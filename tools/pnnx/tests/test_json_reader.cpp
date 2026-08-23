// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "json_reader.h"

#include <stdio.h>
#include <string.h>

#include <limits>
#include <string>

static int test_failures = 0;

static void report_failure(const char* input, const std::string& message)
{
    fprintf(stderr, "json test failed input=%s %s\n", input, message.c_str());
    test_failures++;
}

static bool parse(const char* input, pnnx::JsonValue& value, pnnx::JsonParseError& error)
{
    pnnx::JsonParseOptions options;
    return pnnx::parse_json(input, strlen(input), value, error, options) == 0;
}

static bool parse_bytes(const char* input, size_t input_size, pnnx::JsonValue& value, pnnx::JsonParseError& error)
{
    pnnx::JsonParseOptions options;
    return pnnx::parse_json(input, input_size, value, error, options) == 0;
}

static void expect_type(const char* input, pnnx::JsonType type)
{
    pnnx::JsonValue value;
    pnnx::JsonParseError error;
    if (!parse(input, value, error))
    {
        report_failure(input, error.message);
        return;
    }

    if (value.type() != type)
        report_failure(input, "unexpected type");
}

static void expect_bool(const char* input, bool expected)
{
    pnnx::JsonValue value;
    pnnx::JsonParseError error;
    if (!parse(input, value, error))
    {
        report_failure(input, error.message);
        return;
    }

    if (value.type() != pnnx::JSON_BOOL || value.as_bool() != expected)
        report_failure(input, "unexpected bool");
}

static void expect_int(const char* input, int64_t expected)
{
    pnnx::JsonValue value;
    pnnx::JsonParseError error;
    if (!parse(input, value, error))
    {
        report_failure(input, error.message);
        return;
    }

    if (value.type() != pnnx::JSON_INT64 || value.as_int64() != expected)
        report_failure(input, "unexpected int64");
}

static void expect_uint(const char* input, uint64_t expected)
{
    pnnx::JsonValue value;
    pnnx::JsonParseError error;
    if (!parse(input, value, error))
    {
        report_failure(input, error.message);
        return;
    }

    if (value.type() != pnnx::JSON_UINT64 || value.as_uint64() != expected)
        report_failure(input, "unexpected uint64");
}

static void expect_double(const char* input, double expected)
{
    pnnx::JsonValue value;
    pnnx::JsonParseError error;
    if (!parse(input, value, error))
    {
        report_failure(input, error.message);
        return;
    }

    if (value.type() != pnnx::JSON_DOUBLE || value.as_double() != expected)
        report_failure(input, "unexpected double");
}

static void expect_string(const char* input, const std::string& expected)
{
    pnnx::JsonValue value;
    pnnx::JsonParseError error;
    if (!parse(input, value, error))
    {
        report_failure(input, error.message);
        return;
    }

    if (value.type() != pnnx::JSON_STRING || value.as_string() != expected)
        report_failure(input, "unexpected string");
}

static void expect_string_bytes(const char* input, size_t input_size, const std::string& expected, const char* label)
{
    pnnx::JsonValue value;
    pnnx::JsonParseError error;
    if (!parse_bytes(input, input_size, value, error))
    {
        report_failure(label, error.message);
        return;
    }

    if (value.type() != pnnx::JSON_STRING || value.as_string() != expected)
        report_failure(label, "unexpected string bytes");
}

static void expect_error(const char* input, size_t byte_offset, size_t line, size_t column, const char* message)
{
    pnnx::JsonValue value;
    pnnx::JsonParseError error;
    if (parse(input, value, error))
    {
        report_failure(input, "unexpected success");
        return;
    }

    if (error.byte_offset != byte_offset || error.line != line || error.column != column)
    {
        report_failure(input, "unexpected error position");
        return;
    }

    if (error.message.find(message) == std::string::npos)
        report_failure(input, "unexpected error message " + error.message);
}

static void expect_error_bytes(const char* input, size_t input_size, size_t byte_offset, const char* message, const char* label)
{
    pnnx::JsonValue value;
    pnnx::JsonParseError error;
    if (parse_bytes(input, input_size, value, error))
    {
        report_failure(label, "unexpected success");
        return;
    }

    if (error.byte_offset != byte_offset)
    {
        report_failure(label, "unexpected error position");
        return;
    }

    if (error.message.find(message) == std::string::npos)
        report_failure(label, "unexpected error message " + error.message);
}

static void expect_error_with_options(const char* input, const pnnx::JsonParseOptions& options, size_t byte_offset, const char* message)
{
    pnnx::JsonValue value;
    pnnx::JsonParseError error;
    if (pnnx::parse_json(input, strlen(input), value, error, options) == 0)
    {
        report_failure(input, "unexpected success");
        return;
    }

    if (error.byte_offset != byte_offset)
    {
        report_failure(input, "unexpected error position");
        return;
    }

    if (error.message.find(message) == std::string::npos)
        report_failure(input, "unexpected error message " + error.message);
}

static void expect_composite_values()
{
    const char* input = "[null,true,{\"x\":1}]";
    pnnx::JsonValue value;
    pnnx::JsonParseError error;
    if (!parse(input, value, error))
    {
        report_failure(input, error.message);
        return;
    }

    if (value.type() != pnnx::JSON_ARRAY || value.as_array().size() != 3)
    {
        report_failure(input, "unexpected outer array");
        return;
    }

    if (value.as_array()[0].type() != pnnx::JSON_NULL || value.as_array()[1].type() != pnnx::JSON_BOOL || !value.as_array()[1].as_bool())
    {
        report_failure(input, "unexpected scalar elements");
        return;
    }

    const pnnx::JsonValue& object = value.as_array()[2];
    const pnnx::JsonValue* x = object.find("x");
    if (object.type() != pnnx::JSON_OBJECT || object.as_object().size() != 1 || !x || x->type() != pnnx::JSON_INT64 || x->as_int64() != 1 || object.find("missing"))
        report_failure(input, "unexpected object element");

    const char* unicode_key_input = "{\"\\u4f60\":2}";
    if (!parse(unicode_key_input, value, error))
    {
        report_failure(unicode_key_input, error.message);
        return;
    }

    const pnnx::JsonValue* unicode_key_value = value.find("\xe4\xbd\xa0");
    if (!unicode_key_value || unicode_key_value->type() != pnnx::JSON_INT64 || unicode_key_value->as_int64() != 2)
        report_failure(unicode_key_input, "unicode object key was not decoded");
}

static void expect_empty_containers()
{
    pnnx::JsonValue value;
    pnnx::JsonParseError error;
    if (!parse("[]", value, error) || value.type() != pnnx::JSON_ARRAY || !value.as_array().empty())
        report_failure("[]", error.message.empty() ? "unexpected empty array" : error.message);

    if (!parse("{}", value, error) || value.type() != pnnx::JSON_OBJECT || !value.as_object().empty())
        report_failure("{}", error.message.empty() ? "unexpected empty object" : error.message);
}

static void expect_unicode_strings()
{
    expect_string("\"\\u4f60\\u597d\"", "\xe4\xbd\xa0\xe5\xa5\xbd");
    expect_string("\"\\ud83d\\ude80\"", "\xf0\x9f\x9a\x80");
    expect_string("\"\\ud800\\udc00\"", "\xf0\x90\x80\x80");
    expect_string("\"\\udbff\\udfff\"", "\xf4\x8f\xbf\xbf");
    expect_string("\"\\u00a2\"", "\xc2\xa2");
    expect_string("\"\\u0000\"", std::string(1, '\0'));

    const char raw_utf8[] = {'"', (char)0xc2, (char)0xa2, (char)0xe2, (char)0x82, (char)0xac, (char)0xf0, (char)0x9f, (char)0x9a, (char)0x80, '"'};
    expect_string_bytes(raw_utf8, sizeof(raw_utf8), std::string(raw_utf8 + 1, sizeof(raw_utf8) - 2), "valid raw utf8");
}

static void expect_invalid_utf8()
{
    const char unexpected_continuation[] = {'"', (char)0x80, '"'};
    expect_error_bytes(unexpected_continuation, sizeof(unexpected_continuation), 1, "utf-8", "unexpected utf8 continuation");

    const char overlong_two_byte[] = {'"', (char)0xc0, (char)0xaf, '"'};
    expect_error_bytes(overlong_two_byte, sizeof(overlong_two_byte), 1, "utf-8", "overlong two-byte utf8");

    const char overlong_three_byte[] = {'"', (char)0xe0, (char)0x80, (char)0x80, '"'};
    expect_error_bytes(overlong_three_byte, sizeof(overlong_three_byte), 1, "utf-8", "overlong three-byte utf8");

    const char invalid_continuation[] = {'"', (char)0xe2, '(', (char)0xa1, '"'};
    expect_error_bytes(invalid_continuation, sizeof(invalid_continuation), 2, "continuation", "invalid utf8 continuation");

    const char truncated_sequence[] = {'"', (char)0xe2, (char)0x82};
    expect_error_bytes(truncated_sequence, sizeof(truncated_sequence), 3, "truncated", "truncated utf8");

    const char surrogate_code_point[] = {'"', (char)0xed, (char)0xa0, (char)0x80, '"'};
    expect_error_bytes(surrogate_code_point, sizeof(surrogate_code_point), 1, "surrogate", "utf8 surrogate");

    const char above_unicode_max[] = {'"', (char)0xf4, (char)0x90, (char)0x80, (char)0x80, '"'};
    expect_error_bytes(above_unicode_max, sizeof(above_unicode_max), 1, "range", "utf8 above unicode max");

    const char invalid_five_byte_lead[] = {'"', (char)0xf5, (char)0x80, (char)0x80, (char)0x80, '"'};
    expect_error_bytes(invalid_five_byte_lead, sizeof(invalid_five_byte_lead), 1, "leading", "invalid utf8 leading byte");
}

static void expect_utf8_string_limit()
{
    const char input[] = {'"', (char)0xe2, (char)0x82, (char)0xac, '"'};
    const std::string expected(input + 1, 3);

    pnnx::JsonParseOptions options;
    options.max_string_length = 3;
    pnnx::JsonValue value;
    pnnx::JsonParseError error;
    if (pnnx::parse_json(input, sizeof(input), value, error, options) != 0 || value.type() != pnnx::JSON_STRING || value.as_string() != expected)
    {
        report_failure("utf8 string limit", "unexpected exact-limit rejection");
        return;
    }

    options.max_string_length = 2;
    if (pnnx::parse_json(input, sizeof(input), value, error, options) == 0 || error.byte_offset != 1 || error.message.find("string length limit") == std::string::npos)
        report_failure("utf8 string limit", "unexpected over-limit result");

    const char* escaped_input = "\"\\u20ac\"";
    if (pnnx::parse_json(escaped_input, strlen(escaped_input), value, error, options) == 0 || error.byte_offset != 2 || error.message.find("string length limit") == std::string::npos)
        report_failure("escaped utf8 string limit", "unexpected over-limit result");
}

static void expect_container_limits()
{
    pnnx::JsonParseOptions options;
    options.max_depth = 2;
    expect_error_with_options("[[[]]]", options, 2, "depth limit");

    options = pnnx::JsonParseOptions();
    options.max_nodes = 3;
    expect_error_with_options("[0,1,2]", options, 5, "node limit");

    options = pnnx::JsonParseOptions();
    options.max_nodes = 1;
    pnnx::JsonValue value;
    pnnx::JsonParseError error;
    if (pnnx::parse_json("[]", 2, value, error, options) != 0 || value.type() != pnnx::JSON_ARRAY)
        report_failure("[]", "empty array exceeded one-node budget");

    options = pnnx::JsonParseOptions();
    options.max_depth = 3;
    if (pnnx::parse_json("[[[]]]", 6, value, error, options) != 0)
        report_failure("[[[]]]", "exact depth budget was rejected");

    options = pnnx::JsonParseOptions();
    options.max_nodes = 4;
    if (pnnx::parse_json("[0,1,2]", 7, value, error, options) != 0)
        report_failure("[0,1,2]", "exact node budget was rejected");

    options.max_nodes = 2;
    if (pnnx::parse_json("{\"key\":0}", 9, value, error, options) != 0)
        report_failure("{\"key\":0}", "object key incorrectly consumed node budget");
}

static void expect_string_limit_error()
{
    pnnx::JsonParseOptions options;
    options.max_string_length = 3;

    pnnx::JsonValue value;
    pnnx::JsonParseError error;
    const char* accepted_input = "\"abc\"";
    if (pnnx::parse_json(accepted_input, strlen(accepted_input), value, error, options) != 0 || value.type() != pnnx::JSON_STRING || value.as_string() != "abc")
    {
        report_failure(accepted_input, "unexpected string limit rejection");
        return;
    }

    const char* input = "\"abcd\"";
    if (pnnx::parse_json(input, strlen(input), value, error, options) == 0)
    {
        report_failure(input, "unexpected success");
        return;
    }

    if (error.byte_offset != 4 || error.message.find("string length limit") == std::string::npos)
        report_failure(input, "unexpected string limit error");
}

static void expect_success_resets_outputs()
{
    const char* input = "null";
    pnnx::JsonValue value;
    pnnx::JsonParseError error;
    pnnx::JsonParseOptions options;

    error.byte_offset = 99;
    error.line = 99;
    error.column = 99;
    error.message = "stale";
    if (pnnx::parse_json(input, strlen(input), value, error, options) != 0)
    {
        report_failure(input, "unexpected failure");
        return;
    }

    if (value.type() != pnnx::JSON_NULL || error.byte_offset != 0 || error.line != 1 || error.column != 1 || !error.message.empty())
        report_failure(input, "outputs were not reset");
}

static void expect_node_limit_error()
{
    const char* input = "null";
    pnnx::JsonParseOptions options;
    options.max_nodes = 0;

    pnnx::JsonValue value;
    pnnx::JsonParseError error;
    if (pnnx::parse_json(input, strlen(input), value, error, options) == 0)
    {
        report_failure(input, "unexpected success");
        return;
    }

    if (error.byte_offset != 0 || error.message.find("node limit") == std::string::npos)
        report_failure(input, "unexpected node limit error");
}

static void expect_depth_limit_error()
{
    const char* input = "null";
    pnnx::JsonParseOptions options;
    options.max_depth = 0;

    pnnx::JsonValue value;
    pnnx::JsonParseError error;
    if (pnnx::parse_json(input, strlen(input), value, error, options) == 0)
    {
        report_failure(input, "unexpected success");
        return;
    }

    if (error.byte_offset != 0 || error.message.find("depth limit") == std::string::npos)
        report_failure(input, "unexpected depth limit error");
}

static void expect_embedded_null_error()
{
    const char input[] = {'n', 'u', 'l', 'l', '\0'};

    pnnx::JsonValue value;
    pnnx::JsonParseError error;
    pnnx::JsonParseOptions options;
    if (pnnx::parse_json(input, sizeof(input), value, error, options) == 0)
    {
        report_failure("null\\0", "unexpected success");
        return;
    }

    if (error.byte_offset != 4 || error.line != 1 || error.column != 5 || error.message.find("trailing characters") == std::string::npos)
        report_failure("null\\0", "unexpected embedded null error");
}

static void expect_null_data_error()
{
    pnnx::JsonValue value;
    pnnx::JsonParseError error;
    pnnx::JsonParseOptions options;
    if (pnnx::parse_json(0, 1, value, error, options) == 0)
    {
        report_failure("null data", "unexpected success");
        return;
    }

    if (error.byte_offset != 0 || error.line != 1 || error.column != 1 || error.message.find("data is null") == std::string::npos)
        report_failure("null data", "unexpected null data error");
}

int main()
{
    expect_type("null", pnnx::JSON_NULL);
    expect_bool("true", true);
    expect_bool("false", false);

    expect_int("0", 0);
    expect_int("-0", 0);
    expect_int("9223372036854775807", std::numeric_limits<int64_t>::max());
    expect_int("-9223372036854775808", std::numeric_limits<int64_t>::min());
    expect_uint("9223372036854775808", UINT64_C(9223372036854775808));
    expect_uint("18446744073709551615", std::numeric_limits<uint64_t>::max());

    expect_double("-1.25e+3", -1250.0);
    expect_double("0.5", 0.5);
    expect_double("1E-2", 0.01);

    expect_string("\"\"", "");
    expect_string("\"a\\n\\t\\\"b\"", "a\n\t\"b");
    expect_string("\"\\/\\\\\\b\\f\\r\"", "/\\\b\f\r");

    expect_type(" \t\r\n null \n", pnnx::JSON_NULL);

    expect_error("", 0, 1, 1, "expected json value");
    expect_error("nul", 0, 1, 1, "invalid literal");
    expect_error("True", 0, 1, 1, "expected json value");
    expect_error("+1", 0, 1, 1, "expected json value");
    expect_error("\vnull", 0, 1, 1, "expected json value");
    expect_error("-", 1, 1, 2, "requires digit");
    expect_error("-x", 1, 1, 2, "requires digit");
    expect_error("01", 1, 1, 2, "leading zero");
    expect_error("-01", 2, 1, 3, "leading zero");
    expect_error("18446744073709551616", 20, 1, 21, "integer overflow");
    expect_error("-9223372036854775809", 20, 1, 21, "integer overflow");
    expect_error("1.", 2, 1, 3, "fraction requires digit");
    expect_error("1e", 2, 1, 3, "exponent requires digit");
    expect_error("1e+", 3, 1, 4, "exponent requires digit");
    expect_error("1e9999", 6, 1, 7, "non-finite number");
    expect_error("1 trailing", 2, 1, 3, "trailing characters");
    expect_error("\r\n@", 2, 2, 1, "expected json value");
    expect_error("\r\n  true trailing", 9, 2, 8, "trailing characters");
    expect_error("\"unterminated", 13, 1, 14, "unterminated string");
    expect_error("\"\\x\"", 2, 1, 3, "invalid escape");
    expect_error("\"line\nfeed\"", 5, 1, 6, "control character");
    expect_string_limit_error();
    expect_node_limit_error();
    expect_depth_limit_error();
    expect_embedded_null_error();
    expect_null_data_error();
    expect_success_resets_outputs();

    expect_composite_values();
    expect_empty_containers();
    expect_type(" { \"x\" : [ 1.5, false ] } ", pnnx::JSON_OBJECT);
    expect_unicode_strings();
    expect_invalid_utf8();
    expect_utf8_string_limit();

    expect_error("[", 1, 1, 2, "unterminated array");
    expect_error("[1,]", 3, 1, 4, "trailing comma");
    expect_error("[1 2]", 3, 1, 4, "expected comma or closing bracket");
    expect_error("[\n1,\n]", 5, 3, 1, "trailing comma");
    expect_error("{", 1, 1, 2, "unterminated object");
    expect_error("{:1}", 1, 1, 2, "expected object key");
    expect_error("{\"x\" 1}", 5, 1, 6, "expected colon");
    expect_error("{\"x\":}", 5, 1, 6, "expected json value");
    expect_error("{\"x\":1,}", 7, 1, 8, "trailing comma");
    expect_error("{\"x\":1 \"y\":2}", 7, 1, 8, "expected comma or closing brace");
    expect_error("{\"x\":1", 6, 1, 7, "unterminated object");
    expect_error("{\"x\":1,\"x\":2}", 10, 1, 11, "duplicate key x");
    expect_error("{\"a\":1,\"\\u0061\":2}", 15, 1, 16, "duplicate key a");

    expect_error("\"\\u12xz\"", 5, 1, 6, "invalid unicode escape");
    expect_error("\"\\u12", 5, 1, 6, "incomplete unicode escape");
    expect_error("\"\\ud83d\"", 7, 1, 8, "unpaired high surrogate");
    expect_error("\"\\ude80\"", 2, 1, 3, "unpaired low surrogate");
    expect_error("\"\\ud83d\\u0041\"", 8, 1, 9, "invalid low surrogate");
    expect_error("\"\\ud83dx\"", 7, 1, 8, "unpaired high surrogate");

    expect_container_limits();

    return test_failures == 0 ? 0 : 1;
}
