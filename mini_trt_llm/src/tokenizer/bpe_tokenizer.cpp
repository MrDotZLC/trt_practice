#include "mini_trt_llm/tokenizer/bpe_tokenizer.hpp"

#include "mini_trt_llm/utils/io.hpp"
#include "mini_trt_llm/utils/json.hpp"
#include "mini_trt_llm/utils/logger.hpp"

#include <algorithm>
#include <limits>
#include <sstream>

namespace mini_trt_llm {
namespace {

// 一个码点及其在 UTF-8 里的字节数。非法字节按"单字节码点"处理（等于把原字节原样带走），
// 这样后续的 byte 映射仍然是一一对应的——HF 的 byte-level 路径对非法输入也是这个思路。
struct CodePoint {
    uint32_t value;
    size_t length;
};

CodePoint DecodeUtf8(const std::string& text, size_t offset) {
    const auto byte_at = [&text](size_t i) { return static_cast<uint8_t>(text[i]); };
    const uint8_t first = byte_at(offset);
    if (first < 0x80) return {first, 1};
    size_t extra = 0;
    uint32_t value = 0;
    if ((first & 0xE0) == 0xC0) {
        extra = 1;
        value = first & 0x1F;
    } else if ((first & 0xF0) == 0xE0) {
        extra = 2;
        value = first & 0x0F;
    } else if ((first & 0xF8) == 0xF0) {
        extra = 3;
        value = first & 0x07;
    } else {
        return {first, 1};  // 非法首字节：当作单字节
    }
    if (offset + extra >= text.size()) return {first, 1};
    for (size_t i = 1; i <= extra; ++i) {
        const uint8_t cont = byte_at(offset + i);
        if ((cont & 0xC0) != 0x80) return {first, 1};  // 续字节不合法：当作单字节
        value = (value << 6) | (cont & 0x3F);
    }
    return {value, extra + 1};
}

// 空白表：ASCII 全部 + 常见 Unicode 空白。GPT-2 的 `\s` 是 Unicode 空白，
// 漏掉某个字符会让"空格归谁"整段错位，所以这里按 Unicode Zs/Zl/Zp 与 ASCII 控制空白列全。
bool IsSpaceCodePoint(uint32_t cp) {
    switch (cp) {
        case 0x09: case 0x0A: case 0x0B: case 0x0C: case 0x0D:
        case 0x20: case 0x85: case 0xA0: case 0x1680:
        case 0x2000: case 0x2001: case 0x2002: case 0x2003: case 0x2004: case 0x2005:
        case 0x2006: case 0x2007: case 0x2008: case 0x2009: case 0x200A:
        case 0x2028: case 0x2029: case 0x202F: case 0x205F: case 0x3000:
            return true;
        default:
            return false;
    }
}

// 数字表：ASCII 数字 + 常用非 ASCII 数字（够用即可；漏掉的会被当成"其它"而进入
// `[^\s\p{L}\p{N}]+` 分支，只在极少数语言文本上表现为与 HF 不同）。U+00B2/³/¹ 是 Unicode
// 的 No（数字），HF 的 \p{N} 包含它们，所以这里也要有。
bool IsNumberCodePoint(uint32_t cp) {
    if (cp >= '0' && cp <= '9') return true;
    switch (cp) {
        case 0xB2: case 0xB3: case 0xB9:
            return true;
        default:
            break;
    }
    struct Range { uint32_t lo, hi; };
    static const Range kRanges[] = {
        {0x00BC, 0x00BE}, {0x0660, 0x0669}, {0x06F0, 0x06F9}, {0x0966, 0x096F},
        {0x09E6, 0x09EF}, {0x0A66, 0x0A6F}, {0x0AE6, 0x0AEF}, {0x0B66, 0x0B6F},
        {0x0BE6, 0x0BEF}, {0x0C66, 0x0C6F}, {0x0CE6, 0x0CEF}, {0x0D66, 0x0D6F},
        {0x0E50, 0x0E59}, {0x0ED0, 0x0ED9}, {0x0F20, 0x0F29}, {0x1040, 0x1049},
        {0x17E0, 0x17E9}, {0x1810, 0x1819}, {0xFF10, 0xFF19},
        {0x1D7CE, 0x1D7FF}, {0x1F100, 0x1F10C},
    };
    for (const auto& range : kRanges) {
        if (cp >= range.lo && cp <= range.hi) return true;
    }
    return false;
}

// "其它"（标点 / 符号 / emoji）区间表。**这张表决定"非 ASCII 默认算字母"的对错边界**：
// 落在表内的算"其它"，不落在表内的非 ASCII 一律当字母（中文 / 日文 / 西里尔 / 希腊 / 重音拉丁
// 都在这一侧，它们是 \p{L}）。
bool IsPunctOrSymbolCodePoint(uint32_t cp) {
    if (cp < 0x80) {
        // ASCII 标点与符号：! " # $ % & ' ( ) * + , - . / : ; < = > ? @ [ \ ] ^ _ ` { | } ~
        return !(cp >= '0' && cp <= '9') && !(cp >= 'A' && cp <= 'Z') &&
               !(cp >= 'a' && cp <= 'z') && !IsSpaceCodePoint(cp);
    }
    // Latin-1 里的字母要单独排除：ª(U+00AA) µ(U+00B5) º(U+00BA) 是 \p{L}，
    // 而 ¡¢£¤¥¦§¨©«¬®¯°±´¶·¸»¼½¾¿ 等是标点 / 符号 / 数字。
    switch (cp) {
        case 0xAA: case 0xB5: case 0xBA:
            return false;
        default:
            break;
    }
    struct Range { uint32_t lo, hi; };
    static const Range kRanges[] = {
        {0x00A1, 0x00A9}, {0x00AB, 0x00AC}, {0x00AE, 0x00B1}, {0x00B4, 0x00B4},
        {0x00B6, 0x00B8}, {0x00BB, 0x00BB}, {0x00BF, 0x00BF},
        {0x2000, 0x206F},  // General Punctuation
        {0x2070, 0x209F},  // Super/Subscripts（Σ 等整体归符号）
        {0x20A0, 0x20CF},  // Currency
        {0x2100, 0x214F},  // Letterlike Symbols（含 ™ ℃ 等 So）
        {0x2190, 0x2BFF},  // Arrows / Math / Misc Symbols（含 ☀ ★）
        {0x2E00, 0x2E7F},  // Supplemental Punctuation
        {0x3000, 0x303F},  // CJK Symbols and Punctuation（、。「」等）
        {0xFE10, 0xFE1F}, {0xFE30, 0xFE6F},
        {0xFF01, 0xFF0F}, {0xFF1A, 0xFF20}, {0xFF3B, 0xFF40}, {0xFF5B, 0xFF65},
        {0x1F000, 0x1FAFF},  // 麻将 / 多米诺 / emoji / 符号扩展（🙂 在 0x1F642）
    };
    for (const auto& range : kRanges) {
        if (cp >= range.lo && cp <= range.hi) return true;
    }
    return false;
}

// GPT-2 的 bytes_to_unicode：把 256 个字节映射到"可见且不与空白混淆"的码点，
// 使得任何字节序列都能安全地当作字符串处理（BPE 就是在这些码点上做 merge）。
void BuildByteMaps(std::unordered_map<uint8_t, uint32_t>* byte_to_cp,
                   std::unordered_map<uint32_t, uint8_t>* cp_to_byte) {
    std::vector<uint32_t> bytes;
    for (uint32_t b = '!'; b <= '~'; ++b) bytes.push_back(b);
    for (uint32_t b = 0xA1; b <= 0xAC; ++b) bytes.push_back(b);
    for (uint32_t b = 0xAE; b <= 0xFF; ++b) bytes.push_back(b);
    std::vector<uint32_t> code_points = bytes;
    uint32_t extra = 0;
    for (uint32_t b = 0; b < 256; ++b) {
        if (std::find(bytes.begin(), bytes.end(), b) == bytes.end()) {
            bytes.push_back(b);
            code_points.push_back(256 + extra);
            ++extra;
        }
    }
    for (size_t i = 0; i < bytes.size(); ++i) {
        (*byte_to_cp)[static_cast<uint8_t>(bytes[i])] = code_points[i];
        (*cp_to_byte)[code_points[i]] = static_cast<uint8_t>(bytes[i]);
    }
}

void AppendUtf8(std::string* out, uint32_t cp) {
    if (cp <= 0x7F) {
        out->push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out->push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out->push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out->push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out->push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

// 非法字节的替换码点。HF 的 byte-level 解码遇到不成对的字节会插入 U+FFFD，
// 这里保持同一口径，免得"解码结果看似不同"被误判成实现缺陷。
constexpr uint32_t kReplacementChar = 0xFFFD;

}  // namespace

bool BpeTokenizer::Load(const std::string& vocab_dir) {
    // 去掉尾斜杠：否则会拼出 "dir//vocab.json" 这种只在某些平台才出问题的路径。
    std::string dir = vocab_dir;
    while (dir.size() > 1 && dir.back() == '/') dir.pop_back();
    const std::string vocab_path = dir + "/vocab.json";
    const std::string merges_path = dir + "/merges.txt";

    JsonValue vocab;
    try {
        vocab = LoadJson(vocab_path);
    } catch (const std::exception& exc) {
        MINI_TRT_LOG_ERROR("BPE: 读取 vocab.json 失败：" << vocab_path << "（" << exc.what() << "）");
        return false;
    }
    if (!vocab.IsObject() || vocab.AsObject().empty()) {
        MINI_TRT_LOG_ERROR("BPE: vocab.json 不是非空对象：" << vocab_path);
        return false;
    }

    token_to_id_.clear();
    id_to_token_.clear();
    id_to_token_.resize(vocab.AsObject().size());
    for (const auto& [token, id_value] : vocab.AsObject()) {
        if (!id_value.IsNumber()) {
            MINI_TRT_LOG_ERROR("BPE: vocab.json 的值不是数字：" << token);
            return false;
        }
        const int32_t id = id_value.AsInt();
        if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) {
            MINI_TRT_LOG_ERROR("BPE: vocab.json 的 id 越界：" << token << " -> " << id);
            return false;
        }
        token_to_id_[token] = id;
        id_to_token_[static_cast<size_t>(id)] = token;
    }

    std::vector<char> merges_bytes;
    try {
        merges_bytes = ReadFile(merges_path);
    } catch (const std::exception& exc) {
        MINI_TRT_LOG_ERROR("BPE: 读取 merges.txt 失败：" << merges_path << "（" << exc.what() << "）");
        return false;
    }
    const std::string merges_text(merges_bytes.begin(), merges_bytes.end());
    merge_ranks_.clear();
    std::istringstream stream(merges_text);
    std::string line;
    int32_t rank = 0;
    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        // 首行是 "#version: 0.2" 之类的注释，不是 merge 规则。
        if (line[0] == '#') continue;
        const size_t space = line.find(' ');
        if (space == std::string::npos) {
            MINI_TRT_LOG_ERROR("BPE: merges.txt 的行缺少空格分隔：" << line);
            return false;
        }
        merge_ranks_[line.substr(0, space) + '\x01' + line.substr(space + 1)] = rank++;
    }
    if (merge_ranks_.empty()) {
        MINI_TRT_LOG_ERROR("BPE: merges.txt 没有任何 merge 规则：" << merges_path);
        return false;
    }

    byte_to_codepoint_.clear();
    codepoint_to_byte_.clear();
    BuildByteMaps(&byte_to_codepoint_, &codepoint_to_byte_);

    MINI_TRT_LOG_INFO("BPE: 已加载词表 " << id_to_token_.size() << " 项、merge 规则 "
                                         << merge_ranks_.size() << " 条（" << dir << "）");
    return true;
}

std::vector<std::string> BpeTokenizer::PreTokenizeForTesting(const std::string& text) const {
    return PreTokenize(text);
}

std::vector<std::string> BpeTokenizer::ByteEncodedPiecesForTesting(const std::string& text) const {
    std::vector<std::string> encoded;
    for (const std::string& piece : PreTokenize(text)) {
        encoded.push_back(ByteEncode(piece));
    }
    return encoded;
}

std::vector<std::string> BpeTokenizer::PreTokenize(const std::string& text) const {
    std::vector<std::string> pieces;
    const size_t n = text.size();
    size_t i = 0;

    const auto class_of = [](uint32_t cp) {
        if (IsSpaceCodePoint(cp)) return 0;   // 0 = 空白
        if (IsNumberCodePoint(cp)) return 1;  // 1 = 数字
        if (IsPunctOrSymbolCodePoint(cp)) return 3;  // 3 = 其它（标点 / 符号）
        return 2;                             // 2 = 字母
    };
    const auto is_contraction = [&text, n](size_t pos, size_t* length) {
        static const char* kContractions[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
        if (text[pos] != '\'') return false;
        for (const char* candidate : kContractions) {
            const size_t len = std::char_traits<char>::length(candidate);
            if (pos + len <= n && text.compare(pos, len, candidate) == 0) {
                *length = len;
                return true;
            }
        }
        return false;
    };

    while (i < n) {
        size_t contraction_len = 0;
        if (is_contraction(i, &contraction_len)) {
            pieces.push_back(text.substr(i, contraction_len));
            i += contraction_len;
            continue;
        }

        const CodePoint current = DecodeUtf8(text, i);
        int current_class = class_of(current.value);

        // ` ?\p{L}+` / ` ?\p{N}+` / ` ?[^\s\p{L}\p{N}]+`：可选**一个**前导空格 + 同类连续段。
        // 注意起点：如果当前字符就是空格，是否吃它取决于**下一个字符**是不是字母/数字/其它——
        // 这一条实现时踩过坑（把 " quick" 拆成「空格」+「quick」，见 TROUBLESHOOTING #33）。
        size_t scan = i;
        if (current_class == 0 && i + 1 < n) {
            const CodePoint next = DecodeUtf8(text, i + 1);
            const int next_class = class_of(next.value);
            // 正则里的 ` ?` 只吃**字面空格 U+0020**，不含 tab / 换行 / 不换行空格。
            // 这条踩过两次：写成"任何空白都能当前导空格"会把 "a\tb" 切成 ["a", "\tb"]
            // （HF 是 ["a", "\t", "b"]），见 TROUBLESHOOTING #33。
            if (next_class != 0 && current.value == 0x20) {
                current_class = next_class;  // 这个空格是"前导空格"，跟着后面的同类段一起走
                scan = i + 1;
            }
        }
        if (current_class != 0) {
            // "至多一个"前导空格是关键：两个空格时先由下面的空白分支吃掉一个，剩下的那个才当前导空格
            // （实测 HF：`"a  b"` → ["a", " ", " b"]）。
            while (scan < n) {
                const CodePoint cp = DecodeUtf8(text, scan);
                if (class_of(cp.value) != current_class) break;
                scan += cp.length;
            }
            // scan > i 保证至少吃掉一个同类字符；否则（例如"空格后面不是同类字符"）
            // 落到下面的空白分支，绝不空转。
            if (scan > i) {
                pieces.push_back(text.substr(i, scan - i));
                i = scan;
                continue;
            }
        }

        if (current_class == 0) {
            // `\s+(?!\S)|\s+`：贪婪吃完整段空白，若后面还有非空白则**退一格**——
            // 退出来的那一个空格会紧接着被上面的"可选前导空格"分支吸收（GPT-2 的经典行为，
            // 实测 `"a  b"` → ["a", " ", " b"]）。
            size_t end = i;
            while (end < n) {
                const CodePoint cp = DecodeUtf8(text, end);
                if (!IsSpaceCodePoint(cp.value)) break;
                end += cp.length;
            }
            if (end - i > 1 && end < n) end -= DecodeUtf8(text, end - 1).length;
            pieces.push_back(text.substr(i, end - i));
            i = end;
            continue;
        }

        // 防御性兜底：走到这里理论上不可能，但**循环必须无条件前进**——
        // 死循环的症状比"分词不对"难查得多。
        pieces.push_back(text.substr(i, current.length));
        i += current.length;
    }
    return pieces;
}

std::string BpeTokenizer::ByteEncode(const std::string& raw_bytes) const {
    std::string out;
    out.reserve(raw_bytes.size());
    for (const char raw : raw_bytes) {
        const auto it = byte_to_codepoint_.find(static_cast<uint8_t>(raw));
        // 映射表由 Load 建好；走到这里说明 Load 没成功，属于调用错误。
        if (it == byte_to_codepoint_.end()) return std::string();
        AppendUtf8(&out, it->second);
    }
    return out;
}

std::vector<std::string> BpeTokenizer::ApplyBpe(const std::string& piece) const {
    std::vector<std::string> symbols;
    size_t offset = 0;
    while (offset < piece.size()) {
        const CodePoint cp = DecodeUtf8(piece, offset);
        symbols.push_back(piece.substr(offset, cp.length));
        offset += cp.length;
    }
    if (symbols.size() < 2) return symbols;

    while (symbols.size() > 1) {
        int32_t best_rank = std::numeric_limits<int32_t>::max();
        size_t best_index = 0;
        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            const auto it = merge_ranks_.find(symbols[i] + '\x01' + symbols[i + 1]);
            if (it != merge_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_index = i;
            }
        }
        if (best_rank == std::numeric_limits<int32_t>::max()) break;

        // 合并**所有**该 pair 的出现位置（与 openai/gpt-2 的 encoder 同序：从左往右、不回头），
        // 只合并一处会让长词在后续轮次里与参考分叉。
        const std::string first = symbols[best_index];
        const std::string second = symbols[best_index + 1];
        std::vector<std::string> merged;
        for (size_t i = 0; i < symbols.size();) {
            if (i + 1 < symbols.size() && symbols[i] == first && symbols[i + 1] == second) {
                merged.push_back(first + second);
                i += 2;
            } else {
                merged.push_back(symbols[i]);
                ++i;
            }
        }
        symbols = std::move(merged);
    }
    return symbols;
}

std::vector<int64_t> BpeTokenizer::Encode(const std::string& text) const {
    std::vector<int64_t> ids;
    if (token_to_id_.empty()) {
        MINI_TRT_LOG_ERROR("BPE: Encode 在 Load 成功之前被调用");
        return ids;
    }
    for (const std::string& piece : PreTokenize(text)) {
        const std::string encoded = ByteEncode(piece);
        if (encoded.empty() && !piece.empty()) {
            MINI_TRT_LOG_ERROR("BPE: byte 映射表缺失，Encode 失败");
            return {};
        }
        for (const std::string& symbol : ApplyBpe(encoded)) {
            const auto it = token_to_id_.find(symbol);
            if (it == token_to_id_.end()) {
                // 词表里查不到 = 真故障（不是"跳过就好"）：返回空 vector 让调用方看见失败。
                MINI_TRT_LOG_ERROR("BPE: 词表里没有这个符号，Encode 失败");
                return {};
            }
            ids.push_back(it->second);
        }
    }
    return ids;
}

std::string BpeTokenizer::Decode(const std::vector<int64_t>& ids) const {
    std::string bytes;
    for (const int64_t id : ids) {
        if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) {
            MINI_TRT_LOG_ERROR("BPE: Decode 收到越界 id：" << id);
            continue;
        }
        const std::string& token = id_to_token_[static_cast<size_t>(id)];
        size_t offset = 0;
        while (offset < token.size()) {
            const CodePoint cp = DecodeUtf8(token, offset);
            const auto it = codepoint_to_byte_.find(cp.value);
            if (it == codepoint_to_byte_.end()) {
                bytes.push_back(static_cast<char>(cp.value & 0xFF));
            } else {
                bytes.push_back(static_cast<char>(it->second));
            }
            offset += cp.length;
        }
    }

    // 字节 -> UTF-8：合法序列原样输出，非法字节按 HF 的口径替换成 U+FFFD。
    std::string text;
    size_t i = 0;
    while (i < bytes.size()) {
        const uint8_t first = static_cast<uint8_t>(bytes[i]);
        size_t expected = 0;
        if (first < 0x80) {
            expected = 1;
        } else if ((first & 0xE0) == 0xC0) {
            expected = 2;
        } else if ((first & 0xF0) == 0xE0) {
            expected = 3;
        } else if ((first & 0xF8) == 0xF0) {
            expected = 4;
        }
        bool valid = expected != 0 && i + expected <= bytes.size();
        if (valid) {
            for (size_t k = 1; k < expected; ++k) {
                if ((static_cast<uint8_t>(bytes[i + k]) & 0xC0) != 0x80) {
                    valid = false;
                    break;
                }
            }
        }
        if (valid) {
            text.append(bytes, i, expected);
            i += expected;
        } else {
            AppendUtf8(&text, kReplacementChar);
            ++i;
        }
    }
    return text;
}

}  // namespace mini_trt_llm
