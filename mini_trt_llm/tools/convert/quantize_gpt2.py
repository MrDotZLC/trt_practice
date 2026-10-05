#!/usr/bin/env python3
"""GPT-2 权重的 INT8 离线量化（REQ-017 的路线 C）。

产出（默认落在模型目录）：
  model_int8.safetensors   int8 权重（与源文件**同名同形**，不做任何重排）
  quant_int8.json          量化清单（scale / 粒度 / 轴 / 来源哈希 / 饱和比例）

为什么量化放在 Python 侧（而不是 C++ 建图期）：见
`docs/dev/REQ-017-llm-int8-quant/design.md` 的 D1（路线 C）。要点是
**scale 必须与"被量化的那张张量"同源**；把"算 scale"和"写 int8"放进同一个脚本、
并把来源哈希写进产物，才能把"尺子量 A、裁剪 B"这类错误在**文件级**判出来
（教训见 `docs/PROGRESS.md` §3.0j / `docs/TROUBLESHOOTING.md` #46）。

依赖：只用标准库。有 numpy 时走快路径（大模型必需，124M 参数的纯 Python 会慢到不可用），
没有 numpy 时自动回退到纯 Python——沙箱里没有 numpy，`--self-test` 必须在那儿也能跑。

用法：
  python quantize_gpt2.py --model-dir models/gpt2
  python quantize_gpt2.py --model-dir models/gpt2 --verify
  python quantize_gpt2.py --self-test

模型目录必须带 `config.json` 的 `weight_map`：清单按**建图侧的 TRT 名**选，没有映射就无从
对齐与排除（缺了它脚本会拒绝生成，除非显式加 `--no-config`，那仅适用于 key_prefix 为空的产物）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import tempfile

try:  # 可选依赖：有就走快路径
    import numpy as _np
except Exception:  # pragma: no cover - 沙箱路径
    _np = None

_SAFETENSORS_DTYPE_STR = {
    "F32": "f",
    "F16": "e",
    "I8": "b",
}

# 默认清单里被排除的 2-D 权重：见 design.md 的 D6。
# wpe 只被 gather 读一行，量化它省不到带宽，反而多一条"gather 出 int8 还要 DQ"的路径。
_DEFAULT_EXCLUDE = ("wpe.weight",)

_MANIFEST_NAME = "quant_int8.json"
_INT8_NAME = "model_int8.safetensors"


# ---------------------------------------------------------------------------
# safetensors 读写（只覆盖本脚本需要的 dtype）
# ---------------------------------------------------------------------------


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_safetensors(path: str):
    """返回 (header: dict, data: bytes)。header 形如 {name: {dtype, shape, data_offsets}}。"""
    with open(path, "rb") as handle:
        raw = handle.read()
    if len(raw) < 8:
        raise ValueError(f"{path}: 文件短于 8 字节，不是 safetensors")
    (header_len,) = struct.unpack_from("<Q", raw, 0)
    header_end = 8 + header_len
    if header_end > len(raw):
        raise ValueError(f"{path}: header 长度 {header_len} 越界")
    header = json.loads(raw[8:header_end].decode("utf-8"))
    header.pop("__metadata__", None)
    return header, raw[header_end:]


def _dtype_struct_code(dtype: str) -> str:
    if dtype not in _SAFETENSORS_DTYPE_STR:
        raise ValueError(f"不支持的 dtype {dtype}（本脚本只写 F32 / I8，读还支持 F16）")
    return _SAFETENSORS_DTYPE_STR[dtype]


def _element_count(shape) -> int:
    count = 1
    for dim in shape:
        count *= int(dim)
    return count


def read_tensor_floats(data: bytes, head: dict) -> list:
    """把张量读成 float 列表（纯 Python 表示；numpy 路径在调用点处理）。"""
    code = _dtype_struct_code(head["dtype"])
    begin, end = head["data_offsets"]
    count = _element_count(head["shape"])
    values = list(struct.unpack_from(f"<{count}{code}", data, begin))
    if end - begin != count * struct.calcsize(code):
        raise ValueError("data_offsets 与 shape/dtype 不自洽")
    return values


def write_safetensors(path: str, tensors) -> None:
    """tensors: [(name, dtype, shape, payload_bytes), ...]。

    header 必须是 8 字节对齐（safetensors 的硬要求），所以 JSON 用空格补齐。
    """
    header = {}
    offset = 0
    for name, dtype, shape, payload in tensors:
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        offset += len(payload)

    header_json = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    pad = (-(8 + len(header_json))) % 8
    header_json += b" " * pad

    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(header_json)))
        handle.write(header_json)
        for _, _, _, payload in tensors:
            handle.write(payload)


# ---------------------------------------------------------------------------
# 量化
# ---------------------------------------------------------------------------


def load_weight_map(model_dir: str):
    """读 `config.json` 的 weight_map（TRT 名 → 文件里的 key）；没有就返回 None。

    **为什么清单必须按 TRT 名选，而不是按文件里的 key 选**：建图侧只认 TRT 名（它拿
    weight_map 去查文件），所以"排除 wpe"这类规则必须落在同一个命名空间里。否则一份带
    `transformer.` 前缀的 HF 原始导出会让 `wpe.weight` 这条排除**静默失效**——wpe 被量化，
    而建图侧照样按 TRT 名消费它，两边"哪些张量需要量化"就此对不上。
    """
    path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    weight_map = config.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return None
    return {str(key): str(value) for key, value in weight_map.items()}


def plan_targets(header: dict, weight_map, only=None, exclude=()):
    """定出"量化哪些张量"，返回 [(TRT 名, 文件里的 key)]。

    - 默认策略按 **TRT 名**过滤（2-D 且不在排除名单里）。`--exclude` 是**追加**到默认
      排除名单上的，不是替换——否则一句 `--exclude foo` 就会把默认的 wpe 放回清单。
    - `--only` 是显式点名，优先级高于默认策略，可以给 TRT 名或文件 key；但必须真的存在
      且是 2-D——给错了要响亮失败，不能静默少量化一层。
    """
    pairs = list(weight_map.items()) if weight_map else [(n, n) for n in header]

    if only:
        wanted = set(only)
        targets = []
        for trt_name, source_key in pairs:
            if trt_name not in wanted and source_key not in wanted:
                continue
            if source_key not in header:
                raise KeyError(f"清单指向的 {source_key}（TRT 名 {trt_name}）不在源权重里")
            if len(header[source_key]["shape"]) != 2:
                raise ValueError(
                    f"{trt_name} 不是 2-D 权重（本轮只支持 per-tensor 的 2-D 权重）")
            targets.append((trt_name, source_key))
        missing = wanted - {t for t, _ in targets} - {s for _, s in targets}
        if missing:
            raise KeyError(f"清单里的张量在源权重里找不到：{sorted(missing)}")
        return sorted(targets)

    excluded = set(_DEFAULT_EXCLUDE) | set(exclude)
    targets = []
    for trt_name, source_key in pairs:
        if trt_name in excluded:
            continue
        if source_key not in header:
            raise KeyError(f"weight_map 指向的 {source_key}（TRT 名 {trt_name}）不在源权重里")
        if len(header[source_key]["shape"]) != 2:
            continue
        targets.append((trt_name, source_key))
    return sorted(targets)


def quantize_tensor(values, scale: float):
    """对称量化：q = clamp(round(v/scale), -127, 127)；返回 (codes: list[int], 饱和计数)。"""
    if _np is not None:
        array = _np.asarray(values, dtype=_np.float32)
        scaled = array / _np.float32(scale)
        rounded = _np.rint(scaled)
        saturated = int(_np.count_nonzero(_np.abs(rounded) > 127))
        codes = _np.clip(rounded, -127, 127).astype(_np.int8)
        return codes, saturated

    codes = []
    saturated = 0
    for value in values:
        rounded = int(round(value / scale))
        if abs(rounded) > 127:
            saturated += 1
        codes.append(max(-127, min(127, rounded)))
    return codes, saturated


def _codes_to_bytes(codes) -> bytes:
    if _np is not None:
        return codes.tobytes()
    return struct.pack(f"<{len(codes)}b", *codes)


def _amax(values) -> float:
    if _np is not None:
        return float(_np.max(_np.abs(_np.asarray(values, dtype=_np.float32))))
    return max(abs(float(value)) for value in values)


def quantize(model_dir: str, output_dir: str | None = None, only=None, exclude=None,
             force: bool = False, allow_no_config: bool = False) -> dict:
    output_dir = output_dir or model_dir
    src_path = os.path.join(model_dir, "model.safetensors")
    int8_path = os.path.join(output_dir, _INT8_NAME)
    manifest_path = os.path.join(output_dir, _MANIFEST_NAME)

    if not os.path.isfile(src_path):
        raise FileNotFoundError(f"找不到源权重：{src_path}")
    for path in (int8_path, manifest_path):
        if os.path.exists(path) and not force:
            raise FileExistsError(
                f"{path} 已存在；重新生成会换掉正式产物，需显式加 --force"
            )

    header, data = read_safetensors(src_path)
    weight_map = load_weight_map(model_dir)
    if weight_map is None and not allow_no_config:
        # 没有 weight_map 就无法知道建图侧的 TRT 名，排除名单也就无从生效。
        # 这种时候**拒绝生成**：宁可不产，也不要产出一份命名空间错位的清单。
        raise FileNotFoundError(
            f"找不到 {os.path.join(model_dir, 'config.json')} 的 weight_map：无法确定建图侧的"
            f" TRT 名，清单会写错命名空间。补 config.json，或显式加 --no-config"
            f"（仅适用于 key_prefix 为空的产物）")
    targets = plan_targets(header, weight_map, only=only, exclude=exclude or ())

    entries = []
    tensors = []
    for trt_name, source_key in targets:
        head = header[source_key]
        values = read_tensor_floats(data, head)
        amax = _amax(values)
        # 全零张量给 scale=1，避免除零；量化结果仍是全零。
        scale = (amax / 127.0) if amax > 0.0 else 1.0
        codes, saturated = quantize_tensor(values, scale)
        count = _element_count(head["shape"])
        # int8 产物按**文件里的 key** 命名（建图侧经 weight_map 解析出同一个 key），
        # 清单里同时留 TRT 名——这样"清单 ↔ 产物 ↔ 建图"三者的对应关系可逐条核对。
        tensors.append((source_key, "I8", head["shape"], _codes_to_bytes(codes)))
        entries.append({
            "tensor": trt_name,
            "source_key": source_key,
            "granularity": "per_tensor",
            "axis": None,
            "scales": [scale],
            "saturate_ratio": saturated / count if count else 0.0,
        })

    write_safetensors(int8_path, tensors)

    manifest = {
        "format_version": 1,
        "generator": {
            "tool": "tools/convert/quantize_gpt2.py",
            "command": " ".join(sys.argv),
        },
        "weights_source": {
            "path": os.path.basename(src_path),
            "sha256": _sha256_file(src_path),
        },
        "int8_weights": {
            "path": os.path.basename(int8_path),
            "sha256": _sha256_file(int8_path),
        },
        "scheme": "symmetric_per_tensor",
        "zero_point": 0,
        "entries": entries,
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    return manifest


def verify(model_dir: str, output_dir: str | None = None) -> None:
    """复核产物：① 清单里记的 sha256 与磁盘上的文件一致；② 逐元素重算 int8 码必须相同。

    ① 是 design.md 那句"文件级证据"的兑现——没有它，清单声明的来源就只是自我声明；
    ② 保证产物没被手工改过、也不是用另一份 scale 生成的。
    """
    output_dir = output_dir or model_dir
    manifest_path = os.path.join(output_dir, _MANIFEST_NAME)
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)

    src_path = os.path.join(model_dir, "model.safetensors")
    int8_path = os.path.join(output_dir, _INT8_NAME)
    for label, recorded, actual in (
        ("weights_source", manifest["weights_source"], _sha256_file(src_path)),
        ("int8_weights", manifest["int8_weights"], _sha256_file(int8_path)),
    ):
        if recorded.get("sha256") != actual:
            raise AssertionError(
                f"{label} 的 sha256 与磁盘文件不符：清单 {recorded.get('sha256')} vs 实际 {actual}")

    src_header, src_data = read_safetensors(src_path)
    q_header, q_data = read_safetensors(int8_path)

    for entry in manifest["entries"]:
        source_key = entry["source_key"]
        scale = float(entry["scales"][0])
        if source_key not in q_header:
            raise AssertionError(f"{source_key} 不在 int8 产物里")
        if q_header[source_key]["dtype"] != "I8":
            raise AssertionError(f"{source_key} 的 dtype 不是 I8")
        if list(q_header[source_key]["shape"]) != list(src_header[source_key]["shape"]):
            raise AssertionError(f"{source_key} 的形状与源不一致")

        values = read_tensor_floats(src_data, src_header[source_key])
        expected, _ = quantize_tensor(values, scale)
        begin, end = q_header[source_key]["data_offsets"]
        count = _element_count(src_header[source_key]["shape"])
        actual = struct.unpack_from(f"<{count}b", q_data, begin)
        if end - begin != count:
            raise AssertionError(f"{source_key} 的字节数与形状不符")
        if _np is not None:
            if not _np.array_equal(_np.asarray(expected), _np.asarray(actual)):
                raise AssertionError(f"{source_key} 的 int8 码与重算结果不一致")
        elif list(expected) != list(actual):
            raise AssertionError(f"{source_key} 的 int8 码与重算结果不一致")
    print(f"[OK] verify 通过：sha256 身份 + {len(manifest['entries'])} 个张量的 int8 码逐元素一致")


# ---------------------------------------------------------------------------
# self-test：自造 fixture，把"它会拦人"证明给 ctest 看
# ---------------------------------------------------------------------------


def _make_fixture(model_dir: str, prefix: str = "transformer.") -> None:
    """造一份合成模型目录。

    默认带 `transformer.` 前缀：那正是最容易埋雷的形态——文件里的 key 与建图侧的 TRT 名
    不同名，靠"按 TRT 名排除 wpe"这条规则才判得对。
    """
    os.makedirs(model_dir, exist_ok=True)

    def f32(values):
        return struct.pack(f"<{len(values)}f", *values)

    tensors = [
        # 2-D 权重：默认清单的目标
        (prefix + "h.0.attn.c_attn.weight", "F32", [4, 6],
         f32([(i - 12) * 0.5 for i in range(24)])),
        (prefix + "wte.weight", "F32", [8, 4], f32([0.25 * (i + 1) for i in range(32)])),
        # 排除项：wpe 是 2-D 也在排除名单里；其余是 1-D
        (prefix + "wpe.weight", "F32", [5, 4], f32([1.0] * 20)),
        (prefix + "h.0.attn.c_attn.bias", "F32", [6], f32([0.1] * 6)),
        (prefix + "h.0.ln_1.weight", "F32", [4], f32([1.0] * 4)),
    ]
    write_safetensors(os.path.join(model_dir, "model.safetensors"), tensors)

    config = {
        "model_type": "gpt2",
        "architecture": "decoder_only",
        # weight_map 的方向与 WeightLoader 的约定一致：TRT 侧名字 -> 文件里的 key
        "weight_map": {name[len(prefix):]: name for name, _, _, _ in tensors},
        "source": {"key_prefix": prefix},
    }
    with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = os.path.join(tmp, "gpt2")
        _make_fixture(model_dir)

        manifest = quantize(model_dir)
        trt_names = sorted(entry["tensor"] for entry in manifest["entries"])
        source_keys = sorted(entry["source_key"] for entry in manifest["entries"])
        # TRT 名（建图侧的命名空间，不带前缀）
        assert trt_names == ["h.0.attn.c_attn.weight", "wte.weight"], trt_names
        # 文件 key（带前缀）；**wpe 必须不出现**——尽管它的文件名叫 transformer.wpe.weight
        assert source_keys == ["transformer.h.0.attn.c_attn.weight",
                              "transformer.wte.weight"], source_keys

        # scale 必须等于 amax/127，饱和比例为 0（对称量化下不该有饱和）
        src_header, src_data = read_safetensors(os.path.join(model_dir, "model.safetensors"))
        for entry in manifest["entries"]:
            values = read_tensor_floats(src_data, src_header[entry["source_key"]])
            expected_scale = _amax(values) / 127.0
            assert abs(float(entry["scales"][0]) - expected_scale) < 1e-12, entry
            assert entry["saturate_ratio"] == 0.0, entry
            assert entry["granularity"] == "per_tensor"
            assert entry["axis"] is None
        assert manifest["zero_point"] == 0
        assert manifest["weights_source"]["sha256"] == _sha256_file(
            os.path.join(model_dir, "model.safetensors"))
        assert manifest["int8_weights"]["sha256"] == _sha256_file(
            os.path.join(model_dir, _INT8_NAME))

        # 产物按**文件 key**命名、只含被点名的张量，dtype / shape 不变形
        q_header, _ = read_safetensors(os.path.join(model_dir, _INT8_NAME))
        assert sorted(q_header) == source_keys, sorted(q_header)
        for key in source_keys:
            assert q_header[key]["dtype"] == "I8"
            assert list(q_header[key]["shape"]) == list(src_header[key]["shape"])

        # 逐元素 + 身份复核
        verify(model_dir)

        # 护栏 1：没有 config.json / weight_map 时**拒绝生成**（命名空间无从确定）
        bare_dir = os.path.join(tmp, "bare")
        _make_fixture(bare_dir, prefix="")
        os.remove(os.path.join(bare_dir, "config.json"))
        try:
            quantize(bare_dir)
            raise AssertionError("缺 config.json 没有被拦")
        except FileNotFoundError:
            pass
        # --no-config 是显式逃逸口：key_prefix 为空时（文件 key == TRT 名）语义成立
        fallback = quantize(bare_dir, allow_no_config=True)
        assert sorted(e["tensor"] for e in fallback["entries"]) == [
            "h.0.attn.c_attn.weight", "wte.weight"], fallback["entries"]

        # 护栏 2：已存在的产物必须显式 --force
        try:
            quantize(model_dir)
            raise AssertionError("重复生成没有被拦")
        except FileExistsError:
            pass
        quantize(model_dir, force=True)

        # 护栏 3：清单里点名了不存在的张量 → 拦
        try:
            quantize(model_dir, only=["h.9.attn.c_attn.weight"], force=True)
            raise AssertionError("缺张量没有被拦")
        except KeyError:
            pass

        # 正向：`--only` 用 **TRT 名**点名（不带前缀），产物仍按文件 key 落盘
        only_manifest = quantize(model_dir, only=["wte.weight"], force=True)
        assert [e["tensor"] for e in only_manifest["entries"]] == ["wte.weight"], only_manifest
        assert [e["source_key"] for e in only_manifest["entries"]] == [
            "transformer.wte.weight"], only_manifest

        # 护栏 4：非 2-D 张量 → 拦（本轮只支持 per-tensor 的 2-D 权重）
        try:
            quantize(model_dir, only=["h.0.ln_1.weight"], force=True)
            raise AssertionError("1-D 张量没有被拦")
        except ValueError:
            pass

        # 护栏 5：产物被改坏 → verify 必须拦
        int8_path = os.path.join(model_dir, _INT8_NAME)
        with open(int8_path, "r+b") as handle:
            handle.seek(os.path.getsize(int8_path) - 1)
            last = handle.read(1)
            handle.seek(os.path.getsize(int8_path) - 1)
            handle.write(bytes([(last[0] + 1) % 256]))
        try:
            verify(model_dir)
            raise AssertionError("被改坏的产物没有被 verify 拦住")
        except AssertionError:
            pass

        # 护栏 6：清单里的 sha256 被改 → verify 必须拦（"文件级证据"不是自我声明）
        quantize(model_dir, force=True)
        manifest_path = os.path.join(model_dir, _MANIFEST_NAME)
        with open(manifest_path, encoding="utf-8") as handle:
            tampered = json.load(handle)
        tampered["int8_weights"]["sha256"] = "0" * 64
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(tampered, handle)
        try:
            verify(model_dir)
            raise AssertionError("清单里的假 sha256 没有被 verify 拦住")
        except AssertionError:
            pass

    print("[OK] quantize_gpt2.py self-test：6 道护栏 + 命名空间/数值/身份自检全部通过")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="GPT-2 INT8 离线量化（REQ-017 路线 C）")
    parser.add_argument("--model-dir", help="含 model.safetensors 的模型目录")
    parser.add_argument("--output-dir", help="产物目录（默认与 --model-dir 相同）")
    parser.add_argument("--only", action="append", help="只量化这些张量（可重复）")
    parser.add_argument("--exclude", action="append", help="从默认清单里排除（可重复）")
    parser.add_argument("--granularity", default="per_tensor",
                        help="本轮只支持 per_tensor（per_channel 见 design.md D2）")
    parser.add_argument("--force", action="store_true", help="允许覆盖已存在的产物")
    parser.add_argument("--no-config", action="store_true",
                        help="没有 config.json 的 weight_map 时也允许生成（按文件 key 当 TRT 名，"
                             "仅适用于 key_prefix 为空的产物）")
    parser.add_argument("--verify", action="store_true", help="只做产物复核，不重新量化")
    parser.add_argument("--self-test", action="store_true", help="自造 fixture 跑护栏自检")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    if not args.model_dir:
        parser.error("--model-dir 是必填（或使用 --self-test）")

    if args.granularity != "per_tensor":
        print(f"错误：本轮只支持 per_tensor（收到 {args.granularity}）；"
              f"per_channel 见 docs/dev/REQ-017-llm-int8-quant/design.md 的 D2",
              file=sys.stderr)
        return 2

    try:
        if args.verify:
            verify(args.model_dir, args.output_dir)
            return 0
        manifest = quantize(args.model_dir, args.output_dir, only=args.only,
                            exclude=args.exclude, force=args.force,
                            allow_no_config=args.no_config)
    except Exception as exc:  # 响亮失败：产物可能只写了一半
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    print(f"[OK] 量化 {len(manifest['entries'])} 个张量 → "
          f"{os.path.join(args.output_dir or args.model_dir, _INT8_NAME)} + {_MANIFEST_NAME}")
    print(f"     源 sha256 {manifest['weights_source']['sha256'][:16]}…，"
          f"int8 sha256 {manifest['int8_weights']['sha256'][:16]}…")
    # 饱和比例是"尺子量 A、裁剪 B"最便宜的信号：对称量化的 scale 取自同一张量时它恒为 0，
    # 一旦 >0 就说明算 scale 的张量与写进产物的张量不是同一份（见 docs/PROGRESS.md §3.0j）。
    worst = max((float(entry["saturate_ratio"]) for entry in manifest["entries"]), default=0.0)
    print(f"     饱和比例最大 {worst:.6f}（逐条见清单；对称量化下应恒为 0）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
