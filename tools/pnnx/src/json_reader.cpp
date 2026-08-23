// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "json_reader.h"

#include <string.h>

#include <cmath>
#include <limits>
#include <locale>
#include <sstream>
#include <string>

namespace pnnx {

JsonParseOptions::JsonParseOptions()
{
    max_depth = 256;
    max_nodes = 16 * 1024 * 1024;
    max_string_length = 64 * 1024 * 1024;
}

JsonValue::JsonValue()
{
    type_ = JSON_NULL;
    bool_value_ = false;
    int64_value_ = 0;
    uint64_value_ = 0;
    double_value_ = 0.0;
}

JsonType JsonValue::type() const
{
    return type_;
}

bool JsonValue::as_bool() const
{
    return bool_value_;
}

int64_t JsonValue::as_int64() const
{
    return int64_value_;
}

uint64_t JsonValue::as_uint64() const
{
    return uint64_value_;
}

double JsonValue::as_double() const
{
    return double_value_;
}

const std::string& JsonValue::as_string() const
{
    return string_value_;
}

const std::vector<JsonValue>& JsonValue::as_array() const
{
    return array_value_;
}

const std::map<std::string, JsonValue>& JsonValue::as_object() const
{
    return object_value_;
}

const JsonValue* JsonValue::find(const std::string& key) const
{
    if (type_ != JSON_OBJECT)
        return 0;

    std::map<std::string, JsonValue>::const_iterator it = object_value_.find(key);
    if (it == object_value_.end())
        return 0;

    return &it->second;
}

class JsonParser
{
public:
    JsonParser(const char* data, size_t size, JsonParseError& error, const JsonParseOptions& options)
        : data_(data), size_(size), error_(error), options_(options)
    {
        position_ = 0;
        node_count_ = 0;
    }

    int parse(JsonValue& value)
    {
        error_.byte_offset = 0;
        error_.line = 1;
        error_.column = 1;
        error_.message.clear();

        value = JsonValue();

        if (!data_ && size_ != 0)
        {
            fail(0, "json data is null");
            return -1;
        }

        skip_whitespace();
        if (position_ == size_)
        {
            fail(position_, "expected json value");
            return -1;
        }

        if (!parse_value(value, 1))
            return -1;

        skip_whitespace();
        if (position_ != size_)
        {
            fail(position_, "trailing characters after json value");
            return -1;
        }

        return 0;
    }

private:
    void skip_whitespace()
    {
        while (position_ < size_)
        {
            const char ch = data_[position_];
            if (ch != ' ' && ch != '\t' && ch != '\r' && ch != '\n')
                break;

            position_++;
        }
    }

    bool parse_value(JsonValue& value, size_t depth)
    {
        if (depth > options_.max_depth)
            return fail(position_, "json depth limit exceeded");
        if (node_count_ >= options_.max_nodes)
            return fail(position_, "json node limit exceeded");

        node_count_++;

        const char ch = data_[position_];
        if (ch == 'n')
            return parse_literal("null", value, JSON_NULL, false);
        if (ch == 't')
            return parse_literal("true", value, JSON_BOOL, true);
        if (ch == 'f')
            return parse_literal("false", value, JSON_BOOL, false);
        if (ch == '"')
            return parse_string(value);
        if (ch == '-' || (ch >= '0' && ch <= '9'))
            return parse_number(value);
        if (ch == '[')
            return fail(position_, "json array is not supported");
        if (ch == '{')
            return fail(position_, "json object is not supported");

        return fail(position_, "expected json value");
    }

    bool parse_literal(const char* literal, JsonValue& value, JsonType type, bool bool_value)
    {
        const size_t literal_size = strlen(literal);
        if (literal_size > size_ - position_ || memcmp(data_ + position_, literal, literal_size) != 0)
            return fail(position_, "invalid literal");

        position_ += literal_size;
        value.type_ = type;
        value.bool_value_ = bool_value;
        return true;
    }

    bool append_string_character(std::string& text, char ch, size_t character_offset)
    {
        if (text.size() >= options_.max_string_length)
            return fail(character_offset, "json string length limit exceeded");

        text.push_back(ch);
        return true;
    }

    bool parse_string(JsonValue& value)
    {
        position_++;
        std::string text;

        while (position_ < size_)
        {
            const size_t character_offset = position_;
            const unsigned char ch = (unsigned char)data_[position_++];

            if (ch == '"')
            {
                value.type_ = JSON_STRING;
                value.string_value_.swap(text);
                return true;
            }

            if (ch < 0x20)
                return fail(character_offset, "control character in json string");
            if (ch >= 0x80)
                return fail(character_offset, "non-ascii json string byte is not supported");

            if (ch != '\\')
            {
                if (!append_string_character(text, (char)ch, character_offset))
                    return false;
                continue;
            }

            if (position_ == size_)
                return fail(position_, "unterminated json escape sequence");

            const size_t escape_offset = position_;
            const char escape = data_[position_++];
            char decoded = 0;
            if (escape == '"' || escape == '\\' || escape == '/')
                decoded = escape;
            else if (escape == 'b')
                decoded = '\b';
            else if (escape == 'f')
                decoded = '\f';
            else if (escape == 'n')
                decoded = '\n';
            else if (escape == 'r')
                decoded = '\r';
            else if (escape == 't')
                decoded = '\t';
            else if (escape == 'u')
                return fail(escape_offset, "json unicode escape is not supported");
            else
                return fail(escape_offset, "invalid escape in json string");

            if (!append_string_character(text, decoded, escape_offset))
                return false;
        }

        return fail(position_, "unterminated string");
    }

    bool parse_number(JsonValue& value)
    {
        const size_t number_start = position_;
        bool negative = false;
        if (data_[position_] == '-')
        {
            negative = true;
            position_++;
            if (position_ == size_)
                return fail(position_, "json number requires digit");
        }

        const size_t integer_start = position_;
        if (data_[position_] == '0')
        {
            position_++;
            if (position_ < size_ && data_[position_] >= '0' && data_[position_] <= '9')
                return fail(position_, "leading zero in json number");
        }
        else if (data_[position_] >= '1' && data_[position_] <= '9')
        {
            while (position_ < size_ && data_[position_] >= '0' && data_[position_] <= '9')
                position_++;
        }
        else
        {
            return fail(position_, "json number requires digit");
        }

        bool floating_point = false;
        if (position_ < size_ && data_[position_] == '.')
        {
            floating_point = true;
            position_++;
            if (position_ == size_ || data_[position_] < '0' || data_[position_] > '9')
                return fail(position_, "json number fraction requires digit");

            while (position_ < size_ && data_[position_] >= '0' && data_[position_] <= '9')
                position_++;
        }

        if (position_ < size_ && (data_[position_] == 'e' || data_[position_] == 'E'))
        {
            floating_point = true;
            position_++;
            if (position_ < size_ && (data_[position_] == '+' || data_[position_] == '-'))
                position_++;
            if (position_ == size_ || data_[position_] < '0' || data_[position_] > '9')
                return fail(position_, "json number exponent requires digit");

            while (position_ < size_ && data_[position_] >= '0' && data_[position_] <= '9')
                position_++;
        }

        if (floating_point)
        {
            const std::string number_text(data_ + number_start, position_ - number_start);
            std::istringstream stream(number_text);
            stream.imbue(std::locale::classic());

            double double_value = 0.0;
            stream >> double_value;
            if (!stream || stream.peek() != std::char_traits<char>::eof() || !std::isfinite(double_value))
                return fail(position_, "non-finite number");

            value.type_ = JSON_DOUBLE;
            value.double_value_ = double_value;
            return true;
        }

        uint64_t magnitude = 0;
        bool overflow = false;
        for (size_t i = integer_start; i < position_; i++)
        {
            const uint64_t digit = data_[i] - '0';
            if (magnitude > (std::numeric_limits<uint64_t>::max() - digit) / 10)
            {
                overflow = true;
                break;
            }

            magnitude = magnitude * 10 + digit;
        }

        if (negative)
        {
            const uint64_t int64_min_magnitude = (uint64_t)std::numeric_limits<int64_t>::max() + 1;
            if (overflow || magnitude > int64_min_magnitude)
                return fail(position_, "json integer overflow");

            value.type_ = JSON_INT64;
            if (magnitude == int64_min_magnitude)
                value.int64_value_ = std::numeric_limits<int64_t>::min();
            else
                value.int64_value_ = -(int64_t)magnitude;
            return true;
        }

        if (overflow)
            return fail(position_, "json integer overflow");

        if (magnitude <= (uint64_t)std::numeric_limits<int64_t>::max())
        {
            value.type_ = JSON_INT64;
            value.int64_value_ = (int64_t)magnitude;
        }
        else
        {
            value.type_ = JSON_UINT64;
            value.uint64_value_ = magnitude;
        }

        return true;
    }

    bool fail(size_t byte_offset, const std::string& message)
    {
        if (!error_.message.empty())
            return false;

        if (byte_offset > size_)
            byte_offset = size_;

        error_.byte_offset = byte_offset;
        error_.line = 1;
        error_.column = 1;
        error_.message = message;

        size_t offset = 0;
        while (offset < byte_offset)
        {
            const char ch = data_[offset];
            if (ch == '\r')
            {
                if (offset + 1 < byte_offset && data_[offset + 1] == '\n')
                    offset++;

                error_.line++;
                error_.column = 1;
            }
            else if (ch == '\n')
            {
                error_.line++;
                error_.column = 1;
            }
            else
            {
                error_.column++;
            }

            offset++;
        }

        return false;
    }

private:
    const char* data_;
    size_t size_;
    JsonParseError& error_;
    const JsonParseOptions& options_;
    size_t position_;
    size_t node_count_;
};

int parse_json(const char* data, size_t size, JsonValue& value, JsonParseError& error, const JsonParseOptions& options)
{
    JsonParser parser(data, size, error, options);
    return parser.parse(value);
}

} // namespace pnnx
