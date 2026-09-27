#!/usr/bin/env bash
#
# 性能画像的一键入口（`future_iterations.md` §6.3 与 §11 的 G6）。
#
# 为什么要有这个脚本（而不是在 CMake 里直接堆 nsys 参数）：
#   * 报告名要带时间戳、输出目录要能覆盖、工具缺失要给出人话错误；
#   * 采样前后要记录温度 / 时钟（协议 §11.3 A），拿不到时必须**显式打印"未采集"**；
#   * 被 profile 的进程会打大量日志，必须落文件 + 只回摘要，否则真正的失败信息会被顶掉；
#   * nsys 跑完要顺手导出 kernel / API 摘要并交给 `summarize_nsys.py` 分桶。
#
# 用法：
#   run_profile.sh <nsys|ncu> <basename> <gtest_filter> <binary> [out_dir]
#
# 环境变量：
#   MINI_TRT_PROFILE_OUTPUT_DIR   报告输出目录（默认 /tmp/mini_trt_llm_profiles，**不入库**）
#   MINI_TRT_NCU_LAUNCH_COUNT     ncu 最多采集多少个 kernel（默认 20，避免整网采集过久）
#   MINI_TRT_NCU_KERNEL_FILTER    ncu 的 --kernel-name 过滤（如 "regex:PagedAttention"）
#
# **退出码**：nsys / ncu 会把被 profile 应用的退出码透传出来 → 脚本最后用同一码退出。
# 所以"用例失败"会让 `profile_gpt2` 这个 ninja target 失败，这是**有意的**：失败必须可见。
# 但导出与摘要会先做完、日志摘要也会打印，不需要再去满屏输出里找原因。
#
# 注意：profiling 属 `AGENTS.md` §0.3 的"单测外的测试任务"，执行前须先获批。

set -euo pipefail

usage() {
    echo "用法：run_profile.sh <nsys|ncu> <basename> <gtest_filter> <binary> [out_dir]" >&2
    echo "  gtest_filter 传空字符串表示不限制用例。" >&2
}

if [[ $# -lt 4 || $# -gt 5 ]]; then
    usage
    exit 2
fi

tool="$1"
basename="$2"
gtest_filter="$3"
binary="$4"
out_dir="${5:-${MINI_TRT_PROFILE_OUTPUT_DIR:-/tmp/mini_trt_llm_profiles}}"

case "${tool}" in
    nsys | ncu) ;;
    *)
        echo "未知工具 '${tool}'（只支持 nsys / ncu）" >&2
        usage
        exit 2
        ;;
esac

if [[ ! -x "${binary}" ]]; then
    echo "找不到可执行文件：${binary}（先构建 mini_trt_llm_tests）" >&2
    exit 1
fi

if ! command -v "${tool}" >/dev/null 2>&1; then
    printf '%s\n' \
        "找不到 ${tool}。请确认 CUDA Toolkit 的 profiling 工具已安装并进 PATH。" \
        "（WSL2 上 ncu 若因 performance counter 权限失败，按 AGENTS.md §1 走无头导出，" \
        "  把 .ncu-rep 拷到 Windows 宿主机用 Nsight Compute GUI 打开。）" >&2
    exit 1
fi

mkdir -p "${out_dir}"
stamp="$(date +%Y%m%d_%H%M%S)"
basename_full="${out_dir}/${basename}_${stamp}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
app_log="${basename_full}.app.log"

gpu_state() {
    # 拿不到就打印"未采集"——不能让读者以为它被量过（AGENTS.md §7 的"观测缺口"）。
    local label="$1"
    if command -v nvidia-smi >/dev/null 2>&1; then
        local line
        line="$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw \
            --format=csv,noheader 2>/dev/null || true)"
        echo "[profile] GPU 状态${label} (temp C, sm MHz, power W)=${line:-<无输出>}"
    else
        echo "[profile] GPU 状态${label}：<nvidia-smi 不可用>"
    fi
}

# 只回摘要：被 profile 进程的完整输出在 ${app_log} 里。
print_digest() {
    echo "[profile] ---- 摘要（完整日志：${app_log}）----"
    # 两类都要：① gtest 的结论行与错误行；② **我们自己的报告行**（形如 `[Gpt2DecodePerf] ...`）。
    # 只留 gtest 就看不到测量值——那正是 `profile_*` 存在的意义（第一次跑就漏了，见 #39）。
    # 第 ② 类的模式要求 tag 里**至少有一个小写字母**：这样 `[Gpt2DecodePerf]` / `[SamplerPerf]`
    # 会进来，而 `[INFO]` / `[WARN]` 这类全大写业务日志不会把摘要刷掉（实测吃过一次）。
    grep -aE '^\[  (PASSED|FAILED|SKIPPED)  \]|^\[==========\].*(ran|FAILED|list)|^\[[A-Z][A-Za-z0-9_]*[a-z][A-Za-z0-9_]*\]|does not contain CUDA kernel data|^E[0-9]+ |Error' \
        "${app_log}" 2>/dev/null | tail -40 || true
    echo "[profile] ---- 摘要结束 ----"
}

run_args=()
if [[ -n "${gtest_filter}" ]]; then
    run_args+=("--gtest_filter=${gtest_filter}")
fi

echo "[profile] tool=${tool} basename=${basename} filter='${gtest_filter:-<all>}'"
echo "[profile] binary=${binary}"
echo "[profile] 输出目录=${out_dir}（默认不入库；只保留本机中间产物）"
"${tool}" --version 2>&1 | head -2 || true
gpu_state "(before)"

# 导出某个 nsys stats 报告。
# **注意**：`nsys stats` 自身的 "Generating SQLite..." / "Processing [...]" 走 **stdout**，
# 会和 CSV 混在一起（真机第一次跑就是这么把假 CSV 写出来的）。所以先写临时文件，
# 确认里面真有表头才认；报告不含该类数据时显式报警，而不是留下只有几行消息的假文件。
export_nsys_report() {
    local report_name="$1" out_csv="$2" tmp="${2}.tmp"
    if ! nsys stats --report "${report_name}" --format csv --force-export=true \
            "${report}" > "${tmp}" 2>> "${app_log}"; then
        rm -f "${tmp}"
        echo "[profile] 警告：${report_name} 导出失败（详见 ${app_log}）" >&2
        return 0
    fi
    if grep -q "Total Time" "${tmp}"; then
        mv "${tmp}" "${out_csv}"
        echo "[profile] ${report_name} 摘要：${out_csv}"
    else
        rm -f "${tmp}"
        echo "[profile] 警告：报告不含 ${report_name} 数据（详见 ${app_log}）" >&2
    fi
}

if [[ "${tool}" == "nsys" ]]; then
    # 不加 `--stats=true`：那会把整张统计表打到终端（"日志太多"的主因之一），
    # 而我们要的摘要是下面自己导出的两份 CSV。
    set +e
    nsys profile --force-overwrite=true -o "${basename_full}" \
        "${binary}" "${run_args[@]}" > "${app_log}" 2>&1
    profile_rc=$?
    set -e
    echo "[profile] nsys 退出码=${profile_rc}（nsys 会透传被 profile 应用的退出码）"

    report="${basename_full}.nsys-rep"
    if [[ ! -f "${report}" ]]; then
        echo "[profile] 错误：没有生成 ${report}；见 ${app_log}" >&2
        print_digest
        exit "${profile_rc:-1}"
    fi
    echo "[profile] nsys 报告：${report}"

    export_nsys_report cuda_gpu_kern_sum "${basename_full}.kern_sum.csv"
    export_nsys_report cuda_api_sum "${basename_full}.api_sum.csv"

    if [[ ! -f "${basename_full}.kern_sum.csv" ]]; then
        printf '%s\n' \
            "[profile] 提示：报告里没有 GPU kernel 时间线。WSL2 上这是**已知限制**" \
            "（CUDA API 能采到，GPU kernel 采不到）。**注意**：把这份 .nsys-rep 拷到 Windows" \
            " 也看不到 kernel 时间线——数据压根没被采集，不是查看器的问题（#39 已更正）。" \
            "可行路径见 docs/TROUBLESHOOTING.md + TS-041：① 在 Windows 宿主侧做采集；" \
            "② 不用 profiler，用同一 session 的 SamplerPerf + 本用例取比值。" >&2
    fi

    python_bin=""
    for candidate in python3 python; do
        if command -v "${candidate}" >/dev/null 2>&1; then
            python_bin="${candidate}"
            break
        fi
    done
    if [[ -n "${python_bin}" && -f "${script_dir}/summarize_nsys.py" ]]; then
        if [[ -s "${basename_full}.kern_sum.csv" ]]; then
            "${python_bin}" "${script_dir}/summarize_nsys.py" \
                "${basename_full}.kern_sum.csv" || true
        fi
        if [[ -s "${basename_full}.api_sum.csv" ]]; then
            "${python_bin}" "${script_dir}/summarize_nsys.py" --api \
                "${basename_full}.api_sum.csv" || true
        fi
    else
        echo "[profile] 跳过分桶（找不到 python 或 summarize_nsys.py）" >&2
    fi

    print_digest
    gpu_state "(after)"
    echo "[profile] 完成。报告目录：${out_dir}"
    exit "${profile_rc}"
fi

set +e
ncu_args=(--set full --target-processes all
    --launch-count "${MINI_TRT_NCU_LAUNCH_COUNT:-20}" -o "${basename_full}")
if [[ -n "${MINI_TRT_NCU_KERNEL_FILTER:-}" ]]; then
    ncu_args+=(-k "${MINI_TRT_NCU_KERNEL_FILTER}")
fi
ncu "${ncu_args[@]}" "${binary}" "${run_args[@]}" > "${app_log}" 2>&1
ncu_rc=$?
set -e
echo "[profile] ncu 退出码=${ncu_rc}"
if [[ -f "${basename_full}.ncu-rep" ]]; then
    echo "[profile] ncu 报告：${basename_full}.ncu-rep"
else
    # 区分"权限/计数器不可用"（WSL2 已知限制）与"别的错误"——否则会被当成我们的 bug。
    if grep -aqE "Unknown Error on device|ERR_NVGPUCTRPERM|Permission|not supported" \
            "${app_log}"; then
        echo "[profile] ncu 在 WSL2 上不可用（拿不到 GPU performance counter）——已知限制，" >&2
        echo "[profile]   本机两条 CLI profiling 路径都到不了 kernel 时间线，见 #39 / #41。" >&2
    else
        echo "[profile] 警告：没有生成 .ncu-rep；见 ${app_log}" >&2
    fi
fi
print_digest
gpu_state "(after)"
echo "[profile] 完成。报告目录：${out_dir}"
exit "${ncu_rc}"
