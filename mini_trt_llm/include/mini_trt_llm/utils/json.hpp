#pragma once

#include <cctype>
#include <cstdlib>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace mini_trt_llm {

// Phase 0 极简 JSON 解析器。
// 基于递归下降法实现，支持对象、数组、字符串、数字、bool、null。
// 仅用于加载 model config 等小型配置，不支持 Unicode 转义与浮点指数。
// 后续迭代可替换为 nlohmann/json。

// JSON 值的运行时容器，使用 std::variant 存储具体数据。
// Type 顺序与 variant 的 alternatives 顺序严格一致，GetType 直接通过 index() 映射。
class JsonValue {
 public:
    enum class Type { kNull, kBool, kNumber, kString, kArray, kObject };

    using Object = std::map<std::string, JsonValue>;
    using Array = std::vector<JsonValue>;
    using Value = std::variant<std::monostate, bool, double, std::string,
                               Array, Object>;

    JsonValue() = default;
    explicit JsonValue(bool v) : value_(v) {}
    explicit JsonValue(double v) : value_(v) {}
    explicit JsonValue(int v) : value_(static_cast<double>(v)) {}
    explicit JsonValue(const char* v) : value_(std::string(v)) {}
    explicit JsonValue(const std::string& v) : value_(v) {}
    explicit JsonValue(Array v) : value_(std::move(v)) {}
    explicit JsonValue(Object v) : value_(std::move(v)) {}

    Type GetType() const {
        return static_cast<Type>(value_.index());
    }

    bool IsNull() const { return GetType() == Type::kNull; }
    bool IsBool() const { return GetType() == Type::kBool; }
    bool IsNumber() const { return GetType() == Type::kNumber; }
    bool IsString() const { return GetType() == Type::kString; }
    bool IsArray() const { return GetType() == Type::kArray; }
    bool IsObject() const { return GetType() == Type::kObject; }

    bool AsBool() const { return std::get<bool>(value_); }
    double AsNumber() const { return std::get<double>(value_); }
    int AsInt() const { return static_cast<int>(std::get<double>(value_)); }
    const std::string& AsString() const { return std::get<std::string>(value_); }
    const Array& AsArray() const { return std::get<Array>(value_); }
    Array& AsArray() { return std::get<Array>(value_); }
    const Object& AsObject() const { return std::get<Object>(value_); }
    Object& AsObject() { return std::get<Object>(value_); }

    bool Has(const std::string& key) const {
        if (!IsObject()) return false;
        return AsObject().find(key) != AsObject().end();
    }

    const JsonValue& operator[](const std::string& key) const {
        return AsObject().at(key);
    }

    JsonValue& operator[](const std::string& key) {
        return AsObject()[key];
    }

    const JsonValue& operator[](size_t idx) const {
        return AsArray().at(idx);
    }

    size_t Size() const {
        if (IsArray()) return AsArray().size();
        if (IsObject()) return AsObject().size();
        return 0;
    }

    std::string Dump(int indent = 0) const;

 private:
    Value value_;
};

class JsonParser {
 public:
    // 解析完整 JSON 文本，解析完成后检查是否残留未消费字符。
    JsonValue Parse(const std::string& text);

 private:
    // 根据当前首字符分发到对象 / 数组 / 字符串 / 数字 / 字面量解析器。
    JsonValue ParseValue();

    // 解析 { "key": value, ... }，空对象直接返回。
    JsonValue ParseObject();

    // 解析 [ value, ... ]，空数组直接返回。
    JsonValue ParseArray();

    // 解析双引号字符串，支持标准转义序列；未处理 Unicode \uXXXX。
    JsonValue ParseString();

    // 解析整数 / 小数 / 科学计数法数字字符串，统一用 double 存储。
    JsonValue ParseNumber();
    JsonValue ParseTrue();
    JsonValue ParseFalse();
    JsonValue ParseNull();

    // 跳过空白字符，所有 ParseXxx 入口都先调用以保持位置正确。
    void SkipWhitespace();

    // 查看当前字符，越界返回 '\0'。
    char Peek() const;

    // 消费并返回当前字符，越界时抛出错误。
    char Get();

    // 消费指定字符，若不符则抛出错误。
    void Expect(char c);

    [[noreturn]] void Error(const std::string& msg) const;

    const std::string* text_ = nullptr;
    // 当前解析位置，所有 ParseXxx 函数通过 pos_ 推进。
    size_t pos_ = 0;
};

inline std::string JsonValue::Dump(int indent) const {
    std::ostringstream oss;
    std::string spaces(indent, ' ');
    switch (GetType()) {
        case Type::kNull:
            oss << "null";
            break;
        case Type::kBool:
            oss << (AsBool() ? "true" : "false");
            break;
        case Type::kNumber:
            oss << AsNumber();
            break;
        case Type::kString:
            oss << "\"" << AsString() << "\"";
            break;
        case Type::kArray: {
            oss << "[";
            const auto& arr = AsArray();
            for (size_t i = 0; i < arr.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << arr[i].Dump(indent);
            }
            oss << "]";
            break;
        }
        case Type::kObject: {
            oss << "{";
            const auto& obj = AsObject();
            bool first = true;
            for (const auto& [k, v] : obj) {
                if (!first) oss << ", ";
                first = false;
                oss << "\"" << k << "\": " << v.Dump(indent);
            }
            oss << "}";
            break;
        }
    }
    return oss.str();
}

inline void JsonParser::SkipWhitespace() {
    while (pos_ < text_->size() && std::isspace((*text_)[pos_])) {
        ++pos_;
    }
}

inline char JsonParser::Peek() const {
    if (pos_ >= text_->size()) return '\0';
    return (*text_)[pos_];
}

inline char JsonParser::Get() {
    if (pos_ >= text_->size()) Error("unexpected end of input");
    return (*text_)[pos_++];
}

inline void JsonParser::Expect(char c) {
    if (Get() != c) {
        Error(std::string("expected '") + c + "'");
    }
}

[[noreturn]] inline void JsonParser::Error(const std::string& msg) const {
    std::ostringstream oss;
    oss << "JSON parse error at position " << pos_ << ": " << msg;
    throw std::runtime_error(oss.str());
}

inline JsonValue JsonParser::Parse(const std::string& text) {
    text_ = &text;
    pos_ = 0;
    JsonValue result = ParseValue();
    SkipWhitespace();
    if (pos_ != text.size()) {
        Error("trailing characters");
    }
    return result;
}

inline JsonValue JsonParser::ParseValue() {
    SkipWhitespace();
    char c = Peek();
    switch (c) {
        case '{':
            return ParseObject();
        case '[':
            return ParseArray();
        case '"':
            return ParseString();
        case 't':
            return ParseTrue();
        case 'f':
            return ParseFalse();
        case 'n':
            return ParseNull();
        default:
            if (c == '-' || std::isdigit(c)) {
                return ParseNumber();
            }
            Error(std::string("unexpected character '") + c + "'");
    }
    return JsonValue();  // unreachable
}

inline JsonValue JsonParser::ParseObject() {
    JsonValue::Object obj;
    Expect('{');
    SkipWhitespace();
    if (Peek() == '}') {
        Get();
        return JsonValue(std::move(obj));
    }
    while (true) {
        SkipWhitespace();
        JsonValue key = ParseString();
        SkipWhitespace();
        Expect(':');
        JsonValue value = ParseValue();
        obj[key.AsString()] = std::move(value);
        SkipWhitespace();
        char c = Get();
        if (c == '}') break;
        if (c != ',') Error("expected ',' or '}' in object");
    }
    return JsonValue(std::move(obj));
}

inline JsonValue JsonParser::ParseArray() {
    JsonValue::Array arr;
    Expect('[');
    SkipWhitespace();
    if (Peek() == ']') {
        Get();
        return JsonValue(std::move(arr));
    }
    while (true) {
        arr.push_back(ParseValue());
        SkipWhitespace();
        char c = Get();
        if (c == ']') break;
        if (c != ',') Error("expected ',' or ']' in array");
    }
    return JsonValue(std::move(arr));
}

inline JsonValue JsonParser::ParseString() {
    Expect('"');
    std::string s;
    while (true) {
        char c = Get();
        if (c == '"') break;
        if (c == '\\') {
            char esc = Get();
            switch (esc) {
                case '"': s.push_back('"'); break;
                case '\\': s.push_back('\\'); break;
                case '/': s.push_back('/'); break;
                case 'b': s.push_back('\b'); break;
                case 'f': s.push_back('\f'); break;
                case 'n': s.push_back('\n'); break;
                case 'r': s.push_back('\r'); break;
                case 't': s.push_back('\t'); break;
                default:
                    Error("unknown escape sequence");
            }
        } else {
            s.push_back(c);
        }
    }
    return JsonValue(s);
}

inline JsonValue JsonParser::ParseNumber() {
    size_t start = pos_;
    if (Peek() == '-') Get();
    while (std::isdigit(Peek())) Get();
    if (Peek() == '.') {
        Get();
        while (std::isdigit(Peek())) Get();
    }
    if (Peek() == 'e' || Peek() == 'E') {
        Get();
        if (Peek() == '+' || Peek() == '-') Get();
        while (std::isdigit(Peek())) Get();
    }
    std::string num_str = text_->substr(start, pos_ - start);
    return JsonValue(std::stod(num_str));
}

inline JsonValue JsonParser::ParseTrue() {
    if (text_->compare(pos_, 4, "true") != 0) Error("expected 'true'");
    pos_ += 4;
    return JsonValue(true);
}

inline JsonValue JsonParser::ParseFalse() {
    if (text_->compare(pos_, 5, "false") != 0) Error("expected 'false'");
    pos_ += 5;
    return JsonValue(false);
}

inline JsonValue JsonParser::ParseNull() {
    if (text_->compare(pos_, 4, "null") != 0) Error("expected 'null'");
    pos_ += 4;
    return JsonValue();
}

}  // namespace mini_trt_llm
