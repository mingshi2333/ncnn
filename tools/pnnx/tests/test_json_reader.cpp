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
    expect_error("\"\\u0041\"", 2, 1, 3, "unicode escape is not supported");
    expect_error("[]", 0, 1, 1, "array is not supported");
    expect_error("{}", 0, 1, 1, "object is not supported");

    expect_string_limit_error();
    expect_node_limit_error();
    expect_depth_limit_error();
    expect_embedded_null_error();
    expect_null_data_error();
    expect_success_resets_outputs();

    return test_failures == 0 ? 0 : 1;
}
