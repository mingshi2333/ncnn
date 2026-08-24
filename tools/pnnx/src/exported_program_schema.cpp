// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "exported_program_schema.h"

#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace pnnx {

ExportedProgramHeader::ExportedProgramHeader()
{
    schema_major = 0;
    schema_minor = 0;
}

ExportedSymInt::ExportedSymInt()
{
    type = EXPORTED_SYM_INT_STATIC;
    value = 0;
    has_hint = false;
    hint = 0;
}

ExportedSymInt::ExportedSymInt(int64_t value_)
{
    type = EXPORTED_SYM_INT_STATIC;
    value = value_;
    has_hint = false;
    hint = 0;
}

bool operator==(const ExportedSymInt& a, const ExportedSymInt& b)
{
    return a.type == b.type
           && a.value == b.value
           && a.expression == b.expression
           && a.has_hint == b.has_hint
           && a.hint == b.hint;
}

ExportedSymBool::ExportedSymBool()
{
    is_expression = false;
    value = false;
    has_hint = false;
    hint = false;
}

ExportedSymFloat::ExportedSymFloat()
{
    is_expression = false;
    value = 0.0;
    has_hint = false;
    hint = 0.0;
}

ExportedRangeConstraint::ExportedRangeConstraint()
{
    has_min = false;
    min = 0;
    has_max = false;
    max = 0;
}

ExportedTensorMeta::ExportedTensorMeta()
{
    dtype = 0;
    layout = 0;
    requires_grad = false;
    device_index = 0;
    has_device_index = false;
}

ExportedPayloadEntry::ExportedPayloadEntry()
{
    is_param = false;
    use_pickle = false;
    has_tensor_meta = false;
}

ExportedDevice::ExportedDevice()
{
    index = 0;
    has_index = false;
}

ExportedSymIntListElement::ExportedSymIntListElement()
{
    type = EXPORTED_SYM_INT_LIST_STATIC;
    value = 0;
}

ExportedArgument::ExportedArgument()
{
    type = EXPORTED_ARGUMENT_NONE;
    int_value = 0;
    float_value = 0.0;
    complex_real_value = 0.0;
    complex_imag_value = 0.0;
    bool_value = false;
    enum_value = 0;
}

ExportedNamedArgument::ExportedNamedArgument()
{
    kind = EXPORTED_ARGUMENT_KIND_MISSING;
}

ExportedNode::ExportedNode()
{
    has_name = false;
}

ExportedInputSpec::ExportedInputSpec()
{
    kind = EXPORTED_USER_INPUT;
    persistent = false;
}

ExportedOutputSpec::ExportedOutputSpec()
{
    kind = EXPORTED_USER_OUTPUT;
}

ExportedTreeSpec::ExportedTreeSpec()
{
    type = EXPORTED_TREE_SPEC_NONE;
}

ExportedGraph::ExportedGraph()
{
    is_single_tensor_return = false;
}

static void clear_schema_error(ExportedSchemaError& error)
{
    error.path.clear();
    error.message.clear();
}

static int schema_error(ExportedSchemaError& error, const std::string& path, const std::string& message)
{
    error.path = path;
    error.message = message;
    return -1;
}

static bool is_path_identifier(const std::string& value)
{
    if (value.empty())
        return false;

    const unsigned char first = (unsigned char)value[0];
    if (!((first >= 'a' && first <= 'z') || (first >= 'A' && first <= 'Z') || first == '_'))
        return false;

    for (size_t i = 1; i < value.size(); i++)
    {
        const unsigned char ch = (unsigned char)value[i];
        if (!((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '_'))
            return false;
    }

    return true;
}

static std::string schema_map_key_path(const std::string& path, const std::string& key)
{
    if (is_path_identifier(key))
        return path + "." + key;

    static const char hex_digits[] = "0123456789abcdef";

    std::string result = path + "[\"";
    for (size_t i = 0; i < key.size(); i++)
    {
        const unsigned char ch = (unsigned char)key[i];
        if (ch == '"' || ch == '\\')
        {
            result.push_back('\\');
            result.push_back((char)ch);
        }
        else if (ch < 0x20)
        {
            result += "\\u00";
            result.push_back(hex_digits[ch >> 4]);
            result.push_back(hex_digits[ch & 15]);
        }
        else
        {
            result.push_back((char)ch);
        }
    }
    result += "\"]";

    return result;
}

static const JsonValue* required_field(const JsonValue& object, const std::string& name, const std::string& object_path, ExportedSchemaError& error)
{
    const JsonValue* value = object.find(name);
    if (!value)
    {
        schema_error(error, object_path + "." + name, "missing required field");
        return 0;
    }

    return value;
}

static int read_integer(const JsonValue& value, int64_t& result, const std::string& path, ExportedSchemaError& error)
{
    if (value.type() == JSON_INT64)
    {
        result = value.as_int64();
        return 0;
    }

    if (value.type() == JSON_UINT64)
        return schema_error(error, path, "integer is out of int64 range");

    return schema_error(error, path, "expected integer");
}

static int read_nonnegative_integer(const JsonValue& value, int64_t& result, const std::string& path, const std::string& negative_message, ExportedSchemaError& error)
{
    if (read_integer(value, result, path, error) != 0)
        return -1;

    if (result < 0)
        return schema_error(error, path, negative_message);

    return 0;
}

static int read_string(const JsonValue& value, std::string& result, const std::string& path, ExportedSchemaError& error)
{
    if (value.type() != JSON_STRING)
        return schema_error(error, path, "expected string");

    result = value.as_string();
    return 0;
}

static int read_bool(const JsonValue& value, bool& result, const std::string& path, ExportedSchemaError& error)
{
    if (value.type() != JSON_BOOL)
        return schema_error(error, path, "expected boolean");

    result = value.as_bool();
    return 0;
}

struct TorchProducerVersion
{
    int major;
    int minor;
    int patch;
};

struct VersionCursor
{
    VersionCursor(const std::string& text_value)
        : text(text_value)
    {
        position = 0;
    }

    const std::string& text;
    size_t position;
};

static bool parse_version_component(VersionCursor& cursor, int& value)
{
    if (cursor.position == cursor.text.size() || cursor.text[cursor.position] < '0' || cursor.text[cursor.position] > '9')
        return false;

    int result = 0;
    while (cursor.position < cursor.text.size() && cursor.text[cursor.position] >= '0' && cursor.text[cursor.position] <= '9')
    {
        const int digit = cursor.text[cursor.position] - '0';
        if (result > (std::numeric_limits<int>::max() - digit) / 10)
            return false;

        result = result * 10 + digit;
        cursor.position++;
    }

    value = result;
    return true;
}

static bool parse_torch_version(const std::string& text, TorchProducerVersion& version)
{
    VersionCursor cursor(text);
    if (!parse_version_component(cursor, version.major))
        return false;
    if (cursor.position == text.size() || text[cursor.position] != '.')
        return false;

    cursor.position++;
    if (!parse_version_component(cursor, version.minor))
        return false;

    version.patch = 0;

    if (cursor.position == text.size())
        return true;

    if (text[cursor.position] == '.')
    {
        cursor.position++;
        if (!parse_version_component(cursor, version.patch))
            return false;
    }
    else if (text[cursor.position] == '+' || text[cursor.position] == '-')
    {
        cursor.position++;
        if (cursor.position == text.size())
            return false;
    }
    else
    {
        return false;
    }

    for (; cursor.position < text.size(); cursor.position++)
    {
        const unsigned char ch = (unsigned char)text[cursor.position];
        const bool allowed = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '.' || ch == '+' || ch == '-' || ch == '_';
        if (!allowed)
            return false;
    }

    return true;
}

static int expected_schema_minor(const TorchProducerVersion& version)
{
    if (version.major != 2)
        return -1;
    if (version.minor == 9)
        return 14;
    if (version.minor == 10)
        return 15;
    if (version.minor == 11)
        return 17;
    if (version.minor == 12)
        return 20;
    if (version.minor == 13)
        return 20;

    return -1;
}

int parse_exported_program_header(const JsonValue& value, ExportedProgramHeader& header, ExportedSchemaError& error)
{
    header = ExportedProgramHeader();
    clear_schema_error(error);

    if (value.type() != JSON_OBJECT)
        return schema_error(error, "$", "expected object");

    ExportedProgramHeader parsed_header;

    const JsonValue* schema_version = required_field(value, "schema_version", "$", error);
    if (!schema_version)
        return -1;
    if (schema_version->type() != JSON_OBJECT)
        return schema_error(error, "$.schema_version", "expected object");

    const JsonValue* schema_major = required_field(*schema_version, "major", "$.schema_version", error);
    if (!schema_major)
        return -1;
    const JsonValue* schema_minor = required_field(*schema_version, "minor", "$.schema_version", error);
    if (!schema_minor)
        return -1;

    int64_t schema_major_value = 0;
    int64_t schema_minor_value = 0;
    if (read_nonnegative_integer(*schema_major, schema_major_value, "$.schema_version.major", "schema major must be non-negative", error) != 0)
        return -1;
    if (read_nonnegative_integer(*schema_minor, schema_minor_value, "$.schema_version.minor", "schema minor must be non-negative", error) != 0)
        return -1;
    if (schema_major_value > std::numeric_limits<int>::max())
        return schema_error(error, "$.schema_version.major", "schema major is out of int range");
    if (schema_minor_value > std::numeric_limits<int>::max())
        return schema_error(error, "$.schema_version.minor", "schema minor is out of int range");

    parsed_header.schema_major = (int)schema_major_value;
    parsed_header.schema_minor = (int)schema_minor_value;

    const JsonValue* torch_version = required_field(value, "torch_version", "$", error);
    if (!torch_version)
        return -1;
    if (read_string(*torch_version, parsed_header.torch_version, "$.torch_version", error) != 0)
        return -1;

    TorchProducerVersion torch_version_value;
    if (!parse_torch_version(parsed_header.torch_version, torch_version_value))
        return schema_error(error, "$.torch_version", "invalid torch producer version");

    if (torch_version_value.major < 2 || (torch_version_value.major == 2 && torch_version_value.minor < 8))
        return schema_error(error, "$.torch_version", "legacy exported program producer is unsupported");
    if (torch_version_value.major == 2 && torch_version_value.minor == 8)
        return schema_error(error, "$.torch_version", "PyTorch 2.8 legacy pickled-payload PT2 is unsupported");
    if (torch_version_value.major != 2 || torch_version_value.minor > 13 || (torch_version_value.minor == 13 && torch_version_value.patch > 0) || (torch_version_value.minor == 12 && torch_version_value.patch > 1))
        return schema_error(error, "$.torch_version", "untested torch producer version");

    if (parsed_header.schema_major != 8)
        return schema_error(error, "$.schema_version.major", "incompatible schema major");

    const int required_schema_minor = expected_schema_minor(torch_version_value);
    if (required_schema_minor < 0 || parsed_header.schema_minor != required_schema_minor)
        return schema_error(error, "$.schema_version.minor", "schema minor does not match torch producer");

    const JsonValue* opset_version = required_field(value, "opset_version", "$", error);
    if (!opset_version)
        return -1;
    if (opset_version->type() != JSON_OBJECT)
        return schema_error(error, "$.opset_version", "expected object");

    const std::map<std::string, JsonValue>& opsets = opset_version->as_object();
    for (std::map<std::string, JsonValue>::const_iterator it = opsets.begin(); it != opsets.end(); ++it)
    {
        const std::string opset_path = schema_map_key_path("$.opset_version", it->first);
        int64_t opset = 0;
        if (read_nonnegative_integer(it->second, opset, opset_path, "opset version must be non-negative", error) != 0)
            return -1;

        parsed_header.opset_version[it->first] = opset;
    }

    if (parsed_header.opset_version.find("aten") == parsed_header.opset_version.end())
        return schema_error(error, "$.opset_version.aten", "missing required field");

    header = parsed_header;
    return 0;
}

static int parse_symbolic_expression(const JsonValue& value, std::string& expression, const JsonValue*& hint, const std::string& path, ExportedSchemaError& error)
{
    if (value.type() != JSON_OBJECT)
        return schema_error(error, path, "expected object");

    const JsonValue* expr_str = required_field(value, "expr_str", path, error);
    if (!expr_str)
        return -1;
    if (read_string(*expr_str, expression, path + ".expr_str", error) != 0)
        return -1;
    if (expression.empty())
        return schema_error(error, path + ".expr_str", "symbolic expression must not be empty");

    hint = required_field(value, "hint", path, error);
    if (!hint)
        return -1;

    return 0;
}

static int parse_symint(const JsonValue& value, ExportedSymInt& result, const std::string& path, ExportedSchemaError& error)
{
    result = ExportedSymInt();
    if (value.type() != JSON_OBJECT || value.as_object().size() != 1)
        return schema_error(error, path, "SymInt union must contain exactly one tag");

    const std::map<std::string, JsonValue>::const_iterator it = value.as_object().begin();
    if (it->first == "as_int")
        return read_integer(it->second, result.value, path + ".as_int", error);
    if (it->first != "as_expr")
        return schema_error(error, path, "unknown SymInt tag " + it->first);

    const JsonValue* hint = 0;
    if (parse_symbolic_expression(it->second, result.expression, hint, path + ".as_expr", error) != 0)
        return -1;

    result.type = EXPORTED_SYM_INT_EXPRESSION;
    if (hint->type() == JSON_NULL)
        return 0;
    if (hint->type() != JSON_OBJECT || hint->as_object().size() != 1 || hint->as_object().begin()->first != "as_int")
        return schema_error(error, path + ".as_expr.hint", "SymInt hint must use as_int");
    if (read_integer(hint->as_object().begin()->second, result.hint, path + ".as_expr.hint.as_int", error) != 0)
        return -1;
    result.has_hint = true;
    return 0;
}

static int parse_symbool(const JsonValue& value, ExportedSymBool& result, const std::string& path, ExportedSchemaError& error)
{
    result = ExportedSymBool();
    if (value.type() != JSON_OBJECT || value.as_object().size() != 1)
        return schema_error(error, path, "SymBool union must contain exactly one tag");

    const std::map<std::string, JsonValue>::const_iterator it = value.as_object().begin();
    if (it->first == "as_bool")
        return read_bool(it->second, result.value, path + ".as_bool", error);
    if (it->first != "as_expr")
        return schema_error(error, path, "unknown SymBool tag " + it->first);

    const JsonValue* hint = 0;
    if (parse_symbolic_expression(it->second, result.expression, hint, path + ".as_expr", error) != 0)
        return -1;

    result.is_expression = true;
    if (hint->type() == JSON_NULL)
        return 0;
    if (hint->type() != JSON_OBJECT || hint->as_object().size() != 1 || hint->as_object().begin()->first != "as_bool")
        return schema_error(error, path + ".as_expr.hint", "SymBool hint must use as_bool");
    if (read_bool(hint->as_object().begin()->second, result.hint, path + ".as_expr.hint.as_bool", error) != 0)
        return -1;
    result.has_hint = true;
    return 0;
}

static int read_double(const JsonValue& value, double& result, const std::string& path, ExportedSchemaError& error);

static int parse_symfloat(const JsonValue& value, ExportedSymFloat& result, const std::string& path, ExportedSchemaError& error)
{
    result = ExportedSymFloat();
    if (value.type() != JSON_OBJECT || value.as_object().size() != 1)
        return schema_error(error, path, "SymFloat union must contain exactly one tag");

    const std::map<std::string, JsonValue>::const_iterator it = value.as_object().begin();
    if (it->first == "as_float")
        return read_double(it->second, result.value, path + ".as_float", error);
    if (it->first != "as_expr")
        return schema_error(error, path, "unknown SymFloat tag " + it->first);

    const JsonValue* hint = 0;
    if (parse_symbolic_expression(it->second, result.expression, hint, path + ".as_expr", error) != 0)
        return -1;

    result.is_expression = true;
    if (hint->type() == JSON_NULL)
        return 0;
    if (hint->type() != JSON_OBJECT || hint->as_object().size() != 1 || hint->as_object().begin()->first != "as_float")
        return schema_error(error, path + ".as_expr.hint", "SymFloat hint must use as_float");
    if (read_double(hint->as_object().begin()->second, result.hint, path + ".as_expr.hint.as_float", error) != 0)
        return -1;
    result.has_hint = true;
    return 0;
}

static int parse_symint_array(const JsonValue& value, std::vector<ExportedSymInt>& result, const std::string& path, bool is_size, ExportedSchemaError& error)
{
    if (value.type() != JSON_ARRAY)
        return schema_error(error, path, "expected array");

    const std::vector<JsonValue>& values = value.as_array();
    result.reserve(values.size());
    for (size_t i = 0; i < values.size(); i++)
    {
        std::ostringstream item_path;
        item_path << path << '[' << i << ']';

        ExportedSymInt item;
        if (parse_symint(values[i], item, item_path.str(), error) != 0)
            return -1;
        if (item.type == EXPORTED_SYM_INT_STATIC && item.value < 0)
            return schema_error(error, item_path.str() + ".as_int", is_size ? "tensor size must be non-negative" : "tensor stride must be non-negative");
        if (item.type == EXPORTED_SYM_INT_EXPRESSION && item.has_hint && item.hint < 0)
            return schema_error(error, item_path.str() + ".as_expr.hint.as_int", is_size ? "tensor size hint must be non-negative" : "tensor stride hint must be non-negative");

        result.push_back(item);
    }

    return 0;
}

int parse_exported_tensor_meta(const JsonValue& value, ExportedTensorMeta& tensor_meta, ExportedSchemaError& error, const std::string& path)
{
    tensor_meta = ExportedTensorMeta();
    clear_schema_error(error);

    const std::string tensor_path = path.empty() ? "$" : path;
    if (value.type() != JSON_OBJECT)
        return schema_error(error, tensor_path, "expected object");

    ExportedTensorMeta parsed_meta;

    const JsonValue* dtype = required_field(value, "dtype", tensor_path, error);
    if (!dtype)
        return -1;
    if (read_nonnegative_integer(*dtype, parsed_meta.dtype, tensor_path + ".dtype", "tensor dtype must be non-negative", error) != 0)
        return -1;

    const JsonValue* sizes = required_field(value, "sizes", tensor_path, error);
    if (!sizes)
        return -1;
    if (parse_symint_array(*sizes, parsed_meta.sizes, tensor_path + ".sizes", true, error) != 0)
        return -1;

    const JsonValue* requires_grad = required_field(value, "requires_grad", tensor_path, error);
    if (!requires_grad)
        return -1;
    if (read_bool(*requires_grad, parsed_meta.requires_grad, tensor_path + ".requires_grad", error) != 0)
        return -1;

    const JsonValue* device = required_field(value, "device", tensor_path, error);
    if (!device)
        return -1;
    if (device->type() != JSON_OBJECT)
        return schema_error(error, tensor_path + ".device", "expected object");

    const JsonValue* device_type = required_field(*device, "type", tensor_path + ".device", error);
    if (!device_type)
        return -1;
    if (read_string(*device_type, parsed_meta.device_type, tensor_path + ".device.type", error) != 0)
        return -1;
    if (parsed_meta.device_type.empty())
        return schema_error(error, tensor_path + ".device.type", "device type must not be empty");

    const JsonValue* device_index = required_field(*device, "index", tensor_path + ".device", error);
    if (!device_index)
        return -1;
    if (device_index->type() == JSON_NULL)
    {
        parsed_meta.device_index = 0;
        parsed_meta.has_device_index = false;
    }
    else
    {
        if (read_nonnegative_integer(*device_index, parsed_meta.device_index, tensor_path + ".device.index", "device index must be non-negative", error) != 0)
            return -1;
        parsed_meta.has_device_index = true;
    }

    const JsonValue* strides = required_field(value, "strides", tensor_path, error);
    if (!strides)
        return -1;
    if (parse_symint_array(*strides, parsed_meta.strides, tensor_path + ".strides", false, error) != 0)
        return -1;
    if (parsed_meta.strides.size() != parsed_meta.sizes.size())
        return schema_error(error, tensor_path + ".strides", "tensor stride rank does not match sizes");

    const JsonValue* storage_offset = required_field(value, "storage_offset", tensor_path, error);
    if (!storage_offset)
        return -1;
    if (parse_symint(*storage_offset, parsed_meta.storage_offset, tensor_path + ".storage_offset", error) != 0)
        return -1;
    if (parsed_meta.storage_offset.type == EXPORTED_SYM_INT_STATIC && parsed_meta.storage_offset.value < 0)
        return schema_error(error, tensor_path + ".storage_offset.as_int", "storage offset must be non-negative");
    if (parsed_meta.storage_offset.type == EXPORTED_SYM_INT_EXPRESSION && parsed_meta.storage_offset.has_hint && parsed_meta.storage_offset.hint < 0)
        return schema_error(error, tensor_path + ".storage_offset.as_expr.hint.as_int", "storage offset hint must be non-negative");

    const JsonValue* layout = required_field(value, "layout", tensor_path, error);
    if (!layout)
        return -1;
    if (read_nonnegative_integer(*layout, parsed_meta.layout, tensor_path + ".layout", "tensor layout must be non-negative", error) != 0)
        return -1;

    tensor_meta = parsed_meta;
    return 0;
}

static bool is_single_archive_path_component(const std::string& path_name)
{
    if (path_name.empty() || path_name == "." || path_name == "..")
        return false;

    for (size_t i = 0; i < path_name.size(); i++)
    {
        if (path_name[i] == '/' || path_name[i] == '\\' || path_name[i] == '\0')
            return false;
    }

    return true;
}

int parse_exported_payload_config(const JsonValue& value, ExportedPayloadConfig& payload_config, ExportedSchemaError& error)
{
    payload_config = ExportedPayloadConfig();
    clear_schema_error(error);

    if (value.type() != JSON_OBJECT)
        return schema_error(error, "$", "expected object");

    const JsonValue* config = required_field(value, "config", "$", error);
    if (!config)
        return -1;
    if (config->type() != JSON_OBJECT)
        return schema_error(error, "$.config", "expected object");

    ExportedPayloadConfig parsed_config;
    const std::map<std::string, JsonValue>& entries = config->as_object();
    for (std::map<std::string, JsonValue>::const_iterator it = entries.begin(); it != entries.end(); ++it)
    {
        const std::string entry_path = schema_map_key_path("$.config", it->first);
        if (it->first.empty())
            return schema_error(error, entry_path, "payload name must not be empty");
        if (it->second.type() != JSON_OBJECT)
            return schema_error(error, entry_path, "expected object");

        ExportedPayloadEntry entry;

        const JsonValue* path_name = required_field(it->second, "path_name", entry_path, error);
        if (!path_name)
            return -1;
        if (read_string(*path_name, entry.path_name, entry_path + ".path_name", error) != 0)
            return -1;
        if (!is_single_archive_path_component(entry.path_name))
            return schema_error(error, entry_path + ".path_name", "payload path_name must be a single archive path component");

        const JsonValue* is_param = required_field(it->second, "is_param", entry_path, error);
        if (!is_param)
            return -1;
        if (read_bool(*is_param, entry.is_param, entry_path + ".is_param", error) != 0)
            return -1;

        const JsonValue* use_pickle = required_field(it->second, "use_pickle", entry_path, error);
        if (!use_pickle)
            return -1;
        if (read_bool(*use_pickle, entry.use_pickle, entry_path + ".use_pickle", error) != 0)
            return -1;

        const JsonValue* tensor_meta = required_field(it->second, "tensor_meta", entry_path, error);
        if (!tensor_meta)
            return -1;
        if (tensor_meta->type() == JSON_NULL)
        {
            if (!entry.use_pickle)
                return schema_error(error, entry_path + ".tensor_meta", "non-pickled payload requires tensor metadata");

            entry.has_tensor_meta = false;
        }
        else
        {
            if (parse_exported_tensor_meta(*tensor_meta, entry.tensor_meta, error, entry_path + ".tensor_meta") != 0)
                return -1;

            entry.has_tensor_meta = true;
        }

        parsed_config.entries[it->first] = entry;
    }

    payload_config = parsed_config;
    return 0;
}

static int read_double(const JsonValue& value, double& result, const std::string& path, ExportedSchemaError& error)
{
    if (value.type() == JSON_DOUBLE)
    {
        result = value.as_double();
        return 0;
    }

    if (value.type() == JSON_STRING)
    {
        const std::string& text = value.as_string();
        if (text == "Infinity")
            result = std::numeric_limits<double>::infinity();
        else if (text == "-Infinity")
            result = -std::numeric_limits<double>::infinity();
        else if (text == "NaN")
            result = std::numeric_limits<double>::quiet_NaN();
        else
            return schema_error(error, path, "unknown special float value");

        return 0;
    }

    return schema_error(error, path, "expected float");
}

static int read_device(const JsonValue& value, ExportedDevice& device, const std::string& path, ExportedSchemaError& error)
{
    if (value.type() != JSON_OBJECT)
        return schema_error(error, path, "expected object");

    const JsonValue* type = required_field(value, "type", path, error);
    if (!type)
        return -1;
    if (read_string(*type, device.type, path + ".type", error) != 0)
        return -1;
    if (device.type.empty())
        return schema_error(error, path + ".type", "device type must not be empty");

    const JsonValue* index = required_field(value, "index", path, error);
    if (!index)
        return -1;
    if (index->type() == JSON_NULL)
    {
        device.index = 0;
        device.has_index = false;
        return 0;
    }

    if (read_nonnegative_integer(*index, device.index, path + ".index", "device index must be non-negative", error) != 0)
        return -1;

    device.has_index = true;
    return 0;
}

static int read_tensor_argument(const JsonValue& value, std::string& name, const std::string& path, ExportedSchemaError& error)
{
    if (value.type() != JSON_OBJECT)
        return schema_error(error, path, "expected tensor argument object");

    const JsonValue* name_value = required_field(value, "name", path, error);
    if (!name_value)
        return -1;
    if (read_string(*name_value, name, path + ".name", error) != 0)
        return -1;
    if (name.empty())
        return schema_error(error, path + ".name", "tensor argument name must not be empty");

    return 0;
}

static int read_custom_object_argument(const JsonValue& value, ExportedArgument& argument, const std::string& path, ExportedSchemaError& error)
{
    if (value.type() != JSON_OBJECT)
        return schema_error(error, path, "expected custom object argument");

    const JsonValue* name = required_field(value, "name", path, error);
    if (!name)
        return -1;
    if (read_string(*name, argument.name, path + ".name", error) != 0)
        return -1;
    if (argument.name.empty())
        return schema_error(error, path + ".name", "custom object name must not be empty");

    const JsonValue* class_fqn = required_field(value, "class_fqn", path, error);
    if (!class_fqn)
        return -1;
    if (read_string(*class_fqn, argument.string_value, path + ".class_fqn", error) != 0)
        return -1;
    if (argument.string_value.empty())
        return schema_error(error, path + ".class_fqn", "custom object class_fqn must not be empty");

    argument.type = EXPORTED_ARGUMENT_UNSUPPORTED;
    argument.unsupported_tag = "as_custom_obj";
    return 0;
}

static int read_integer_array(const JsonValue& value, std::vector<int64_t>& result, const std::string& path, ExportedSchemaError& error)
{
    if (value.type() != JSON_ARRAY)
        return schema_error(error, path, "expected array");

    const std::vector<JsonValue>& values = value.as_array();
    result.reserve(values.size());
    for (size_t i = 0; i < values.size(); i++)
    {
        std::ostringstream item_path;
        item_path << path << '[' << i << ']';

        int64_t item = 0;
        if (read_integer(values[i], item, item_path.str(), error) != 0)
            return -1;
        result.push_back(item);
    }

    return 0;
}

static int read_double_array(const JsonValue& value, std::vector<double>& result, const std::string& path, ExportedSchemaError& error)
{
    if (value.type() != JSON_ARRAY)
        return schema_error(error, path, "expected array");

    const std::vector<JsonValue>& values = value.as_array();
    result.reserve(values.size());
    for (size_t i = 0; i < values.size(); i++)
    {
        std::ostringstream item_path;
        item_path << path << '[' << i << ']';

        double item = 0.0;
        if (read_double(values[i], item, item_path.str(), error) != 0)
            return -1;
        result.push_back(item);
    }

    return 0;
}

static int read_bool_array(const JsonValue& value, std::vector<bool>& result, const std::string& path, ExportedSchemaError& error)
{
    if (value.type() != JSON_ARRAY)
        return schema_error(error, path, "expected array");

    const std::vector<JsonValue>& values = value.as_array();
    result.reserve(values.size());
    for (size_t i = 0; i < values.size(); i++)
    {
        std::ostringstream item_path;
        item_path << path << '[' << i << ']';

        bool item = false;
        if (read_bool(values[i], item, item_path.str(), error) != 0)
            return -1;
        result.push_back(item);
    }

    return 0;
}

static int read_string_array(const JsonValue& value, std::vector<std::string>& result, const std::string& path, ExportedSchemaError& error)
{
    if (value.type() != JSON_ARRAY)
        return schema_error(error, path, "expected array");

    const std::vector<JsonValue>& values = value.as_array();
    result.reserve(values.size());
    for (size_t i = 0; i < values.size(); i++)
    {
        std::ostringstream item_path;
        item_path << path << '[' << i << ']';

        std::string item;
        if (read_string(values[i], item, item_path.str(), error) != 0)
            return -1;
        result.push_back(item);
    }

    return 0;
}

static int read_tensor_argument_array(const JsonValue& value, std::vector<std::string>& result, const std::string& path, ExportedSchemaError& error)
{
    if (value.type() != JSON_ARRAY)
        return schema_error(error, path, "expected array");

    const std::vector<JsonValue>& values = value.as_array();
    result.reserve(values.size());
    for (size_t i = 0; i < values.size(); i++)
    {
        std::ostringstream item_path;
        item_path << path << '[' << i << ']';

        std::string name;
        if (read_tensor_argument(values[i], name, item_path.str(), error) != 0)
            return -1;
        result.push_back(name);
    }

    return 0;
}

static int read_static_sym_argument(const JsonValue& value, const std::string& static_tag, const std::string& dynamic_tag, JsonType static_type, ExportedArgument& argument, const std::string& path, ExportedSchemaError& error)
{
    if (value.type() != JSON_OBJECT || value.as_object().size() != 1)
        return schema_error(error, path, "symbolic argument union must contain exactly one tag");

    const std::map<std::string, JsonValue>::const_iterator it = value.as_object().begin();
    if (it->first == dynamic_tag)
    {
        if (read_string(it->second, argument.name, path + "." + dynamic_tag, error) != 0)
            return -1;
        if (argument.name.empty())
            return schema_error(error, path + "." + dynamic_tag, "symbolic argument name must not be empty");
        argument.type = EXPORTED_ARGUMENT_UNSUPPORTED;
        return 0;
    }
    if (it->first != static_tag)
        return schema_error(error, path, "unknown symbolic argument tag " + it->first);
    if (static_type == JSON_DOUBLE)
    {
        double result = 0.0;
        return read_double(it->second, result, path + "." + static_tag, error);
    }
    if (it->second.type() != static_type)
    {
        if (static_type == JSON_INT64)
            return schema_error(error, path + "." + static_tag, "expected integer");
        return schema_error(error, path + "." + static_tag, "expected boolean");
    }

    return 0;
}

static int parse_exported_graph(const JsonValue& value, ExportedGraph& graph, const std::string& path, ExportedSchemaError& error);

static int parse_exported_argument_value(const JsonValue& value, ExportedArgument& argument, const std::string& path, ExportedSchemaError& error)
{
    argument = ExportedArgument();

    if (value.type() != JSON_OBJECT || value.as_object().size() != 1)
        return schema_error(error, path, "argument union must contain exactly one tag");

    const std::map<std::string, JsonValue>::const_iterator it = value.as_object().begin();
    const std::string& tag = it->first;
    const JsonValue& payload = it->second;
    const std::string tag_path = path + "." + tag;

    if (tag == "as_none")
    {
        if (payload.type() != JSON_BOOL)
            return schema_error(error, tag_path, "expected boolean");
        if (!payload.as_bool())
            return schema_error(error, tag_path, "as_none must be true");
        argument.type = EXPORTED_ARGUMENT_NONE;
        return 0;
    }
    if (tag == "as_tensor")
    {
        argument.type = EXPORTED_ARGUMENT_TENSOR;
        return read_tensor_argument(payload, argument.name, tag_path, error);
    }
    if (tag == "as_tensors")
    {
        argument.type = EXPORTED_ARGUMENT_TENSOR_LIST;
        return read_tensor_argument_array(payload, argument.tensor_names, tag_path, error);
    }
    if (tag == "as_int")
    {
        argument.type = EXPORTED_ARGUMENT_INT;
        return read_integer(payload, argument.int_value, tag_path, error);
    }
    if (tag == "as_ints")
    {
        argument.type = EXPORTED_ARGUMENT_INT_LIST;
        return read_integer_array(payload, argument.int_values, tag_path, error);
    }
    if (tag == "as_float")
    {
        argument.type = EXPORTED_ARGUMENT_FLOAT;
        return read_double(payload, argument.float_value, tag_path, error);
    }
    if (tag == "as_floats")
    {
        argument.type = EXPORTED_ARGUMENT_FLOAT_LIST;
        return read_double_array(payload, argument.float_values, tag_path, error);
    }
    if (tag == "as_complex")
    {
        if (payload.type() != JSON_OBJECT)
            return schema_error(error, tag_path, "expected object");

        const JsonValue* real = required_field(payload, "real", tag_path, error);
        if (!real)
            return -1;
        if (read_double(*real, argument.complex_real_value, tag_path + ".real", error) != 0)
            return -1;

        const JsonValue* imag = required_field(payload, "imag", tag_path, error);
        if (!imag)
            return -1;
        if (read_double(*imag, argument.complex_imag_value, tag_path + ".imag", error) != 0)
            return -1;

        argument.type = EXPORTED_ARGUMENT_COMPLEX;
        return 0;
    }
    if (tag == "as_string")
    {
        argument.type = EXPORTED_ARGUMENT_STRING;
        return read_string(payload, argument.string_value, tag_path, error);
    }
    if (tag == "as_strings")
    {
        argument.type = EXPORTED_ARGUMENT_STRING_LIST;
        return read_string_array(payload, argument.string_values, tag_path, error);
    }
    if (tag == "as_bool")
    {
        argument.type = EXPORTED_ARGUMENT_BOOL;
        return read_bool(payload, argument.bool_value, tag_path, error);
    }
    if (tag == "as_bools")
    {
        argument.type = EXPORTED_ARGUMENT_BOOL_LIST;
        return read_bool_array(payload, argument.bool_values, tag_path, error);
    }
    if (tag == "as_scalar_type" || tag == "as_memory_format" || tag == "as_layout")
    {
        if (read_nonnegative_integer(payload, argument.enum_value, tag_path, "enum value must be non-negative", error) != 0)
            return -1;
        argument.type = tag == "as_scalar_type" ? EXPORTED_ARGUMENT_SCALAR_TYPE : tag == "as_memory_format" ? EXPORTED_ARGUMENT_MEMORY_FORMAT : EXPORTED_ARGUMENT_LAYOUT;
        return 0;
    }
    if (tag == "as_device")
    {
        argument.type = EXPORTED_ARGUMENT_DEVICE;
        return read_device(payload, argument.device_value, tag_path, error);
    }
    if (tag == "as_graph")
    {
        if (payload.type() != JSON_OBJECT)
            return schema_error(error, tag_path, "expected object");

        const JsonValue* name = required_field(payload, "name", tag_path, error);
        if (!name)
            return -1;
        if (read_string(*name, argument.graph_name, tag_path + ".name", error) != 0)
            return -1;
        if (argument.graph_name.empty())
            return schema_error(error, tag_path + ".name", "graph name must not be empty");

        const JsonValue* graph = required_field(payload, "graph", tag_path, error);
        if (!graph)
            return -1;

        argument.graph_value.reset(new ExportedGraph);
        if (parse_exported_graph(*graph, *argument.graph_value, tag_path + ".graph", error) != 0)
            return -1;

        argument.type = EXPORTED_ARGUMENT_GRAPH;
        return 0;
    }
    if (tag == "as_sym_int" || tag == "as_sym_float" || tag == "as_sym_bool")
    {
        const std::string static_tag = tag == "as_sym_int" ? "as_int" : tag == "as_sym_float" ? "as_float" : "as_bool";
        const JsonType static_type = tag == "as_sym_int" ? JSON_INT64 : tag == "as_sym_float" ? JSON_DOUBLE : JSON_BOOL;
        if (read_static_sym_argument(payload, static_tag, "as_name", static_type, argument, tag_path, error) != 0)
            return -1;
        if (argument.type == EXPORTED_ARGUMENT_UNSUPPORTED)
        {
            argument.type = tag == "as_sym_int" ? EXPORTED_ARGUMENT_SYMBOLIC_INT : tag == "as_sym_float" ? EXPORTED_ARGUMENT_SYMBOLIC_FLOAT : EXPORTED_ARGUMENT_SYMBOLIC_BOOL;
            return 0;
        }

        const JsonValue& static_value = payload.as_object().begin()->second;
        if (tag == "as_sym_int")
        {
            argument.type = EXPORTED_ARGUMENT_INT;
            argument.int_value = static_value.as_int64();
        }
        else if (tag == "as_sym_float")
        {
            argument.type = EXPORTED_ARGUMENT_FLOAT;
            if (read_double(static_value, argument.float_value, tag_path + ".as_float", error) != 0)
                return -1;
        }
        else
        {
            argument.type = EXPORTED_ARGUMENT_BOOL;
            argument.bool_value = static_value.as_bool();
        }
        return 0;
    }
    if (tag == "as_sym_ints")
    {
        if (payload.type() != JSON_ARRAY)
            return schema_error(error, tag_path, "expected array");

        bool dynamic = false;
        std::vector<ExportedSymIntListElement> elements;
        const std::vector<JsonValue>& values = payload.as_array();
        elements.reserve(values.size());
        for (size_t i = 0; i < values.size(); i++)
        {
            std::ostringstream item_path;
            item_path << tag_path << '[' << i << ']';

            ExportedArgument item;
            if (read_static_sym_argument(values[i], "as_int", "as_name", JSON_INT64, item, item_path.str(), error) != 0)
                return -1;

            ExportedSymIntListElement element;
            if (item.type == EXPORTED_ARGUMENT_UNSUPPORTED)
            {
                dynamic = true;
                element.type = EXPORTED_SYM_INT_LIST_SYMBOLIC;
                element.name = item.name;
            }
            else
            {
                element.value = values[i].as_object().begin()->second.as_int64();
            }
            elements.push_back(element);
        }

        if (dynamic)
        {
            argument.type = EXPORTED_ARGUMENT_SYMBOLIC_INT_LIST;
            argument.symbolic_int_values.swap(elements);
        }
        else
        {
            argument.type = EXPORTED_ARGUMENT_INT_LIST;
            argument.int_values.reserve(elements.size());
            for (size_t i = 0; i < elements.size(); i++)
                argument.int_values.push_back(elements[i].value);
        }
        return 0;
    }
    if (tag == "as_sym_floats" || tag == "as_sym_bools")
    {
        if (payload.type() != JSON_ARRAY)
            return schema_error(error, tag_path, "expected array");

        const std::string static_tag = tag == "as_sym_floats" ? "as_float" : "as_bool";
        const JsonType static_type = tag == "as_sym_floats" ? JSON_DOUBLE : JSON_BOOL;
        bool dynamic = false;
        const std::vector<JsonValue>& values = payload.as_array();
        for (size_t i = 0; i < values.size(); i++)
        {
            std::ostringstream item_path;
            item_path << tag_path << '[' << i << ']';

            ExportedArgument item;
            if (read_static_sym_argument(values[i], static_tag, "as_name", static_type, item, item_path.str(), error) != 0)
                return -1;
            if (item.type == EXPORTED_ARGUMENT_UNSUPPORTED)
            {
                dynamic = true;
                if (argument.name.empty())
                    argument.name = item.name;
                continue;
            }

            const JsonValue& static_value = values[i].as_object().begin()->second;
            if (tag == "as_sym_floats")
            {
                double item = 0.0;
                if (read_double(static_value, item, item_path.str() + ".as_float", error) != 0)
                    return -1;
                argument.float_values.push_back(item);
            }
            else
                argument.bool_values.push_back(static_value.as_bool());
        }

        if (dynamic)
        {
            argument.type = EXPORTED_ARGUMENT_UNSUPPORTED;
            argument.unsupported_tag = tag;
        }
        else
        {
            argument.type = tag == "as_sym_floats" ? EXPORTED_ARGUMENT_FLOAT_LIST : EXPORTED_ARGUMENT_BOOL_LIST;
        }
        return 0;
    }
    if (tag == "as_custom_obj")
        return read_custom_object_argument(payload, argument, tag_path, error);

    JsonType expected_type = JSON_NULL;
    bool known_unsupported = true;
    if (tag == "as_optional_tensor" || tag == "as_string_to_argument")
        expected_type = JSON_OBJECT;
    else if (tag == "as_optional_tensors" || tag == "as_nested_tensors" || tag == "as_int_lists" || tag == "as_float_lists")
        expected_type = JSON_ARRAY;
    else if (tag == "as_operator")
        expected_type = JSON_STRING;
    else
        known_unsupported = false;

    if (!known_unsupported)
        return schema_error(error, path, "unknown argument tag " + tag);
    if (payload.type() != expected_type)
        return schema_error(error, tag_path, "invalid payload type for unsupported argument");

    argument.type = EXPORTED_ARGUMENT_UNSUPPORTED;
    argument.unsupported_tag = tag;
    if (tag == "as_operator")
        argument.string_value = payload.as_string();
    return 0;
}

static int parse_exported_argument_array(const JsonValue& value, std::vector<ExportedArgument>& arguments, const std::string& path, ExportedSchemaError& error)
{
    if (value.type() != JSON_ARRAY)
        return schema_error(error, path, "expected array");

    const std::vector<JsonValue>& values = value.as_array();
    arguments.reserve(values.size());
    for (size_t i = 0; i < values.size(); i++)
    {
        std::ostringstream item_path;
        item_path << path << '[' << i << ']';

        ExportedArgument argument;
        if (parse_exported_argument_value(values[i], argument, item_path.str(), error) != 0)
            return -1;
        arguments.push_back(argument);
    }

    return 0;
}

static int validate_string_map(const JsonValue& value, const std::string& path, ExportedSchemaError& error)
{
    if (value.type() != JSON_OBJECT)
        return schema_error(error, path, "expected object");

    const std::map<std::string, JsonValue>& entries = value.as_object();
    for (std::map<std::string, JsonValue>::const_iterator it = entries.begin(); it != entries.end(); ++it)
    {
        if (it->second.type() != JSON_STRING)
            return schema_error(error, schema_map_key_path(path, it->first), "expected string");
    }

    return 0;
}

static int parse_named_argument(const JsonValue& value, ExportedNamedArgument& named_argument, const std::string& path, ExportedSchemaError& error)
{
    if (value.type() != JSON_OBJECT)
        return schema_error(error, path, "expected object");

    const JsonValue* name = required_field(value, "name", path, error);
    if (!name)
        return -1;
    if (read_string(*name, named_argument.name, path + ".name", error) != 0)
        return -1;

    const JsonValue* argument = required_field(value, "arg", path, error);
    if (!argument)
        return -1;
    if (parse_exported_argument_value(*argument, named_argument.arg, path + ".arg", error) != 0)
        return -1;

    const JsonValue* kind = value.find("kind");
    if (!kind || kind->type() == JSON_NULL)
    {
        named_argument.kind = EXPORTED_ARGUMENT_KIND_MISSING;
    }
    else
    {
        int64_t kind_value = 0;
        if (read_integer(*kind, kind_value, path + ".kind", error) != 0)
            return -1;
        if (kind_value < EXPORTED_ARGUMENT_KIND_UNKNOWN || kind_value > EXPORTED_ARGUMENT_KIND_KEYWORD)
            return schema_error(error, path + ".kind", "unknown argument kind");

        named_argument.kind = (ExportedArgumentKind)kind_value;
    }

    if (named_argument.name.empty() && named_argument.kind != EXPORTED_ARGUMENT_KIND_POSITIONAL)
        return schema_error(error, path + ".name", "named argument name must not be empty");

    return 0;
}

static int parse_named_argument_array(const JsonValue& value, std::vector<ExportedNamedArgument>& arguments, const std::string& path, ExportedSchemaError& error)
{
    if (value.type() != JSON_ARRAY)
        return schema_error(error, path, "expected array");

    const std::vector<JsonValue>& values = value.as_array();
    arguments.reserve(values.size());
    for (size_t i = 0; i < values.size(); i++)
    {
        std::ostringstream item_path;
        item_path << path << '[' << i << ']';

        ExportedNamedArgument argument;
        if (parse_named_argument(values[i], argument, item_path.str(), error) != 0)
            return -1;
        arguments.push_back(argument);
    }

    return 0;
}

static int parse_exported_node(const JsonValue& value, ExportedNode& node, const std::string& path, ExportedSchemaError& error)
{
    if (value.type() != JSON_OBJECT)
        return schema_error(error, path, "expected object");

    const JsonValue* target = required_field(value, "target", path, error);
    if (!target)
        return -1;
    if (read_string(*target, node.target, path + ".target", error) != 0)
        return -1;
    if (node.target.empty())
        return schema_error(error, path + ".target", "node target must not be empty");

    const JsonValue* inputs = required_field(value, "inputs", path, error);
    if (!inputs)
        return -1;
    if (parse_named_argument_array(*inputs, node.inputs, path + ".inputs", error) != 0)
        return -1;

    const JsonValue* outputs = required_field(value, "outputs", path, error);
    if (!outputs)
        return -1;
    if (parse_exported_argument_array(*outputs, node.outputs, path + ".outputs", error) != 0)
        return -1;

    const JsonValue* metadata = required_field(value, "metadata", path, error);
    if (!metadata)
        return -1;
    if (validate_string_map(*metadata, path + ".metadata", error) != 0)
        return -1;

    const JsonValue* is_hop_single_tensor_return = value.find("is_hop_single_tensor_return");
    if (is_hop_single_tensor_return && is_hop_single_tensor_return->type() != JSON_NULL && is_hop_single_tensor_return->type() != JSON_BOOL)
        return schema_error(error, path + ".is_hop_single_tensor_return", "expected boolean or null");
    if (is_hop_single_tensor_return && is_hop_single_tensor_return->type() == JSON_BOOL && is_hop_single_tensor_return->as_bool())
        return schema_error(error, path + ".is_hop_single_tensor_return", "higher-order node return is unsupported");

    const JsonValue* name = value.find("name");
    if (name && name->type() != JSON_NULL)
    {
        if (read_string(*name, node.name, path + ".name", error) != 0)
            return -1;
        if (node.name.empty())
            return schema_error(error, path + ".name", "node name must not be empty");
        node.has_name = true;
    }

    return 0;
}

static int parse_exported_node_array(const JsonValue& value, std::vector<ExportedNode>& nodes, const std::string& path, ExportedSchemaError& error)
{
    if (value.type() != JSON_ARRAY)
        return schema_error(error, path, "expected array");

    const std::vector<JsonValue>& values = value.as_array();
    nodes.reserve(values.size());
    for (size_t i = 0; i < values.size(); i++)
    {
        std::ostringstream item_path;
        item_path << path << '[' << i << ']';

        ExportedNode node;
        if (parse_exported_node(values[i], node, item_path.str(), error) != 0)
            return -1;
        nodes.push_back(node);
    }

    return 0;
}

static int validate_tensor_argument(const ExportedArgument& argument, const std::map<std::string, ExportedTensorMeta>& tensor_values, const std::string& tensor_values_path, ExportedSchemaError& error)
{
    if (argument.type == EXPORTED_ARGUMENT_TENSOR)
    {
        if (tensor_values.find(argument.name) == tensor_values.end())
            return schema_error(error, schema_map_key_path(tensor_values_path, argument.name), "missing tensor metadata");
        return 0;
    }

    if (argument.type == EXPORTED_ARGUMENT_TENSOR_LIST)
    {
        for (size_t i = 0; i < argument.tensor_names.size(); i++)
        {
            if (tensor_values.find(argument.tensor_names[i]) == tensor_values.end())
                return schema_error(error, schema_map_key_path(tensor_values_path, argument.tensor_names[i]), "missing tensor metadata");
        }
    }

    return 0;
}

static int validate_symbol_argument(const ExportedArgument& argument, const ExportedGraph& graph, const std::string& graph_path, ExportedSchemaError& error)
{
    if (argument.type == EXPORTED_ARGUMENT_SYMBOLIC_INT && graph.sym_int_values.find(argument.name) == graph.sym_int_values.end())
        return schema_error(error, schema_map_key_path(graph_path + ".sym_int_values", argument.name), "missing symbolic int value");
    if (argument.type == EXPORTED_ARGUMENT_SYMBOLIC_BOOL && graph.sym_bool_values.find(argument.name) == graph.sym_bool_values.end())
        return schema_error(error, schema_map_key_path(graph_path + ".sym_bool_values", argument.name), "missing symbolic bool value");
    if (argument.type == EXPORTED_ARGUMENT_SYMBOLIC_FLOAT && graph.sym_float_values.find(argument.name) == graph.sym_float_values.end())
        return schema_error(error, schema_map_key_path(graph_path + ".sym_float_values", argument.name), "missing symbolic float value");
    if (argument.type == EXPORTED_ARGUMENT_SYMBOLIC_INT_LIST)
    {
        for (size_t i = 0; i < argument.symbolic_int_values.size(); i++)
        {
            const ExportedSymIntListElement& item = argument.symbolic_int_values[i];
            if (item.type == EXPORTED_SYM_INT_LIST_SYMBOLIC && graph.sym_int_values.find(item.name) == graph.sym_int_values.end())
                return schema_error(error, schema_map_key_path(graph_path + ".sym_int_values", item.name), "missing symbolic int value");
        }
    }

    return 0;
}

static int validate_graph_arguments(const ExportedGraph& graph, const std::string& graph_path, ExportedSchemaError& error)
{
    const std::string tensor_values_path = graph_path + ".tensor_values";
    for (size_t i = 0; i < graph.inputs.size(); i++)
    {
        if (validate_tensor_argument(graph.inputs[i], graph.tensor_values, tensor_values_path, error) != 0)
            return -1;
        if (validate_symbol_argument(graph.inputs[i], graph, graph_path, error) != 0)
            return -1;
    }
    for (size_t i = 0; i < graph.outputs.size(); i++)
    {
        if (validate_tensor_argument(graph.outputs[i], graph.tensor_values, tensor_values_path, error) != 0)
            return -1;
        if (validate_symbol_argument(graph.outputs[i], graph, graph_path, error) != 0)
            return -1;
    }
    for (size_t i = 0; i < graph.nodes.size(); i++)
    {
        for (size_t j = 0; j < graph.nodes[i].inputs.size(); j++)
        {
            if (validate_tensor_argument(graph.nodes[i].inputs[j].arg, graph.tensor_values, tensor_values_path, error) != 0)
                return -1;
            if (validate_symbol_argument(graph.nodes[i].inputs[j].arg, graph, graph_path, error) != 0)
                return -1;
        }
        for (size_t j = 0; j < graph.nodes[i].outputs.size(); j++)
        {
            if (validate_tensor_argument(graph.nodes[i].outputs[j], graph.tensor_values, tensor_values_path, error) != 0)
                return -1;
            if (validate_symbol_argument(graph.nodes[i].outputs[j], graph, graph_path, error) != 0)
                return -1;
        }
    }

    return 0;
}

template<typename T>
static int parse_symbol_map(const JsonValue& value, std::map<std::string, T>& result, const std::string& path, int (*parse_value)(const JsonValue&, T&, const std::string&, ExportedSchemaError&), ExportedSchemaError& error)
{
    if (value.type() != JSON_OBJECT)
        return schema_error(error, path, "expected object");

    const std::map<std::string, JsonValue>& entries = value.as_object();
    for (std::map<std::string, JsonValue>::const_iterator it = entries.begin(); it != entries.end(); ++it)
    {
        const std::string item_path = schema_map_key_path(path, it->first);
        if (it->first.empty())
            return schema_error(error, item_path, "symbolic value name must not be empty");

        T parsed;
        if (parse_value(it->second, parsed, item_path, error) != 0)
            return -1;
        result[it->first] = parsed;
    }

    return 0;
}

static int parse_exported_graph(const JsonValue& value, ExportedGraph& graph, const std::string& path, ExportedSchemaError& error)
{
    if (value.type() != JSON_OBJECT)
        return schema_error(error, path, "expected object");

    const JsonValue* inputs = required_field(value, "inputs", path, error);
    if (!inputs)
        return -1;
    if (parse_exported_argument_array(*inputs, graph.inputs, path + ".inputs", error) != 0)
        return -1;

    const JsonValue* outputs = required_field(value, "outputs", path, error);
    if (!outputs)
        return -1;
    if (parse_exported_argument_array(*outputs, graph.outputs, path + ".outputs", error) != 0)
        return -1;

    const JsonValue* nodes = required_field(value, "nodes", path, error);
    if (!nodes)
        return -1;
    if (parse_exported_node_array(*nodes, graph.nodes, path + ".nodes", error) != 0)
        return -1;

    const JsonValue* tensor_values = required_field(value, "tensor_values", path, error);
    if (!tensor_values)
        return -1;
    if (tensor_values->type() != JSON_OBJECT)
        return schema_error(error, path + ".tensor_values", "expected object");

    const std::map<std::string, JsonValue>& tensor_entries = tensor_values->as_object();
    for (std::map<std::string, JsonValue>::const_iterator it = tensor_entries.begin(); it != tensor_entries.end(); ++it)
    {
        const std::string tensor_path = schema_map_key_path(path + ".tensor_values", it->first);
        if (it->first.empty())
            return schema_error(error, tensor_path, "tensor value name must not be empty");

        ExportedTensorMeta meta;
        if (parse_exported_tensor_meta(it->second, meta, error, tensor_path) != 0)
            return -1;
        graph.tensor_values[it->first] = meta;
    }

    const JsonValue* sym_int_values = required_field(value, "sym_int_values", path, error);
    if (!sym_int_values)
        return -1;
    if (parse_symbol_map(*sym_int_values, graph.sym_int_values, path + ".sym_int_values", parse_symint, error) != 0)
        return -1;

    const JsonValue* sym_bool_values = required_field(value, "sym_bool_values", path, error);
    if (!sym_bool_values)
        return -1;
    if (parse_symbol_map(*sym_bool_values, graph.sym_bool_values, path + ".sym_bool_values", parse_symbool, error) != 0)
        return -1;

    const JsonValue* sym_float_values = value.find("sym_float_values");
    if (sym_float_values && parse_symbol_map(*sym_float_values, graph.sym_float_values, path + ".sym_float_values", parse_symfloat, error) != 0)
        return -1;

    const JsonValue* is_single_tensor_return = value.find("is_single_tensor_return");
    if (is_single_tensor_return)
    {
        if (read_bool(*is_single_tensor_return, graph.is_single_tensor_return, path + ".is_single_tensor_return", error) != 0)
            return -1;
        if (graph.is_single_tensor_return)
            return schema_error(error, path + ".is_single_tensor_return", "higher-order single tensor graph is unsupported");
    }

    const JsonValue* custom_obj_values = value.find("custom_obj_values");
    if (custom_obj_values)
    {
        if (custom_obj_values->type() != JSON_OBJECT)
            return schema_error(error, path + ".custom_obj_values", "expected object");

        const std::map<std::string, JsonValue>& custom_entries = custom_obj_values->as_object();
        for (std::map<std::string, JsonValue>::const_iterator it = custom_entries.begin(); it != custom_entries.end(); ++it)
        {
            const std::string custom_path = schema_map_key_path(path + ".custom_obj_values", it->first);
            if (it->first.empty())
                return schema_error(error, custom_path, "custom object value name must not be empty");

            ExportedArgument argument;
            if (read_custom_object_argument(it->second, argument, custom_path, error) != 0)
                return -1;
            graph.custom_obj_values[it->first] = argument;
        }
    }

    return validate_graph_arguments(graph, path, error);
}

static int read_nonempty_string_field(const JsonValue& object, const std::string& name, std::string& result, const std::string& path, ExportedSchemaError& error)
{
    const JsonValue* value = required_field(object, name, path, error);
    if (!value)
        return -1;
    if (read_string(*value, result, path + "." + name, error) != 0)
        return -1;
    if (result.empty())
        return schema_error(error, path + "." + name, name + " must not be empty");

    return 0;
}

static int parse_constant_value(const JsonValue& value, ExportedArgument& argument, const std::string& path, ExportedSchemaError& error)
{
    argument = ExportedArgument();

    if (value.type() != JSON_OBJECT || value.as_object().size() != 1)
        return schema_error(error, path, "constant value union must contain exactly one tag");

    const std::map<std::string, JsonValue>::const_iterator it = value.as_object().begin();
    const std::string tag_path = path + "." + it->first;
    if (it->first == "as_none")
    {
        if (it->second.type() != JSON_BOOL)
            return schema_error(error, tag_path, "expected boolean");
        if (!it->second.as_bool())
            return schema_error(error, tag_path, "as_none must be true");
        argument.type = EXPORTED_ARGUMENT_NONE;
        return 0;
    }
    if (it->first == "as_int")
    {
        argument.type = EXPORTED_ARGUMENT_INT;
        return read_integer(it->second, argument.int_value, tag_path, error);
    }
    if (it->first == "as_float")
    {
        argument.type = EXPORTED_ARGUMENT_FLOAT;
        return read_double(it->second, argument.float_value, tag_path, error);
    }
    if (it->first == "as_string")
    {
        argument.type = EXPORTED_ARGUMENT_STRING;
        return read_string(it->second, argument.string_value, tag_path, error);
    }
    if (it->first == "as_bool")
    {
        argument.type = EXPORTED_ARGUMENT_BOOL;
        return read_bool(it->second, argument.bool_value, tag_path, error);
    }

    return schema_error(error, path, "unknown constant value tag " + it->first);
}

static int parse_input_spec(const JsonValue& value, ExportedInputSpec& spec, const std::string& path, ExportedSchemaError& error)
{
    if (value.type() != JSON_OBJECT || value.as_object().size() != 1)
        return schema_error(error, path, "input spec union must contain exactly one tag");

    const std::map<std::string, JsonValue>::const_iterator it = value.as_object().begin();
    const std::string& tag = it->first;
    const JsonValue& payload = it->second;
    const std::string tag_path = path + "." + tag;
    if (payload.type() != JSON_OBJECT)
        return schema_error(error, tag_path, "expected object");

    if (tag == "user_input")
    {
        const JsonValue* argument = required_field(payload, "arg", tag_path, error);
        if (!argument)
            return -1;
        if (parse_exported_argument_value(*argument, spec.arg, tag_path + ".arg", error) != 0)
            return -1;
        spec.kind = EXPORTED_USER_INPUT;
        return 0;
    }
    if (tag == "constant_input")
    {
        if (read_nonempty_string_field(payload, "name", spec.arg.name, tag_path, error) != 0)
            return -1;
        const std::string input_name = spec.arg.name;

        const JsonValue* constant_value = required_field(payload, "value", tag_path, error);
        if (!constant_value)
            return -1;
        if (parse_constant_value(*constant_value, spec.arg, tag_path + ".value", error) != 0)
            return -1;
        spec.arg.name = input_name;
        spec.kind = EXPORTED_CONSTANT_INPUT;
        return 0;
    }
    if (tag == "parameter" || tag == "buffer" || tag == "tensor_constant")
    {
        const JsonValue* argument = required_field(payload, "arg", tag_path, error);
        if (!argument)
            return -1;
        spec.arg.type = EXPORTED_ARGUMENT_TENSOR;
        if (read_tensor_argument(*argument, spec.arg.name, tag_path + ".arg", error) != 0)
            return -1;

        const std::string target_field = tag == "parameter" ? "parameter_name" : tag == "buffer" ? "buffer_name" : "tensor_constant_name";
        if (read_nonempty_string_field(payload, target_field, spec.target, tag_path, error) != 0)
            return -1;

        if (tag == "parameter")
            spec.kind = EXPORTED_PARAMETER;
        else if (tag == "buffer")
        {
            spec.kind = EXPORTED_BUFFER;
            const JsonValue* persistent = required_field(payload, "persistent", tag_path, error);
            if (!persistent)
                return -1;
            if (read_bool(*persistent, spec.persistent, tag_path + ".persistent", error) != 0)
                return -1;
        }
        else
            spec.kind = EXPORTED_TENSOR_CONSTANT;

        return 0;
    }
    if (tag == "custom_obj")
    {
        const JsonValue* argument = required_field(payload, "arg", tag_path, error);
        if (!argument)
            return -1;
        if (read_custom_object_argument(*argument, spec.arg, tag_path + ".arg", error) != 0)
            return -1;
        if (read_nonempty_string_field(payload, "custom_obj_name", spec.target, tag_path, error) != 0)
            return -1;
        spec.kind = EXPORTED_CUSTOM_OBJ;
        return 0;
    }
    if (tag == "token")
    {
        const JsonValue* argument = required_field(payload, "arg", tag_path, error);
        if (!argument)
            return -1;
        if (read_tensor_argument(*argument, spec.arg.name, tag_path + ".arg", error) != 0)
            return -1;
        spec.arg.type = EXPORTED_ARGUMENT_UNSUPPORTED;
        spec.arg.unsupported_tag = "token";
        spec.kind = EXPORTED_TOKEN;
        return 0;
    }

    return schema_error(error, path, "unknown input spec tag " + tag);
}

static int parse_input_specs(const JsonValue& value, std::vector<ExportedInputSpec>& specs, const std::string& path, ExportedSchemaError& error)
{
    if (value.type() != JSON_ARRAY)
        return schema_error(error, path, "expected array");

    const std::vector<JsonValue>& values = value.as_array();
    specs.reserve(values.size());
    for (size_t i = 0; i < values.size(); i++)
    {
        std::ostringstream item_path;
        item_path << path << '[' << i << ']';

        ExportedInputSpec spec;
        if (parse_input_spec(values[i], spec, item_path.str(), error) != 0)
            return -1;
        specs.push_back(spec);
    }

    return 0;
}

static int parse_output_spec(const JsonValue& value, ExportedOutputSpec& spec, const std::string& path, ExportedSchemaError& error)
{
    if (value.type() != JSON_OBJECT || value.as_object().size() != 1)
        return schema_error(error, path, "output spec union must contain exactly one tag");

    const std::map<std::string, JsonValue>::const_iterator it = value.as_object().begin();
    const std::string& tag = it->first;
    const JsonValue& payload = it->second;
    const std::string tag_path = path + "." + tag;
    if (payload.type() != JSON_OBJECT)
        return schema_error(error, tag_path, "expected object");

    if (tag == "user_output")
    {
        const JsonValue* argument = required_field(payload, "arg", tag_path, error);
        if (!argument)
            return -1;
        if (parse_exported_argument_value(*argument, spec.arg, tag_path + ".arg", error) != 0)
            return -1;
        spec.kind = EXPORTED_USER_OUTPUT;
        return 0;
    }
    if (tag == "token")
    {
        const JsonValue* argument = required_field(payload, "arg", tag_path, error);
        if (!argument)
            return -1;
        if (read_tensor_argument(*argument, spec.arg.name, tag_path + ".arg", error) != 0)
            return -1;
        spec.arg.type = EXPORTED_ARGUMENT_UNSUPPORTED;
        spec.arg.unsupported_tag = "token";
        spec.kind = EXPORTED_OUTPUT_TOKEN;
        return 0;
    }

    const bool is_tensor_output = tag == "loss_output" || tag == "buffer_mutation" || tag == "parameter_mutation"
                                  || tag == "gradient_to_parameter" || tag == "gradient_to_user_input" || tag == "user_input_mutation";
    if (!is_tensor_output)
        return schema_error(error, path, "unknown output spec tag " + tag);

    const JsonValue* argument = required_field(payload, "arg", tag_path, error);
    if (!argument)
        return -1;
    spec.arg.type = EXPORTED_ARGUMENT_TENSOR;
    if (read_tensor_argument(*argument, spec.arg.name, tag_path + ".arg", error) != 0)
        return -1;

    std::string target_field;
    if (tag == "loss_output")
        spec.kind = EXPORTED_LOSS_OUTPUT;
    else if (tag == "buffer_mutation")
    {
        spec.kind = EXPORTED_BUFFER_MUTATION;
        target_field = "buffer_name";
    }
    else if (tag == "parameter_mutation")
    {
        spec.kind = EXPORTED_PARAMETER_MUTATION;
        target_field = "parameter_name";
    }
    else if (tag == "gradient_to_parameter")
    {
        spec.kind = EXPORTED_GRADIENT_TO_PARAMETER;
        target_field = "parameter_name";
    }
    else if (tag == "gradient_to_user_input")
    {
        spec.kind = EXPORTED_GRADIENT_TO_USER_INPUT;
        target_field = "user_input_name";
    }
    else if (tag == "user_input_mutation")
    {
        spec.kind = EXPORTED_USER_INPUT_MUTATION;
        target_field = "user_input_name";
    }
    if (!target_field.empty() && read_nonempty_string_field(payload, target_field, spec.target, tag_path, error) != 0)
        return -1;

    return 0;
}

static int parse_output_specs(const JsonValue& value, std::vector<ExportedOutputSpec>& specs, const std::string& path, ExportedSchemaError& error)
{
    if (value.type() != JSON_ARRAY)
        return schema_error(error, path, "expected array");

    const std::vector<JsonValue>& values = value.as_array();
    specs.reserve(values.size());
    for (size_t i = 0; i < values.size(); i++)
    {
        std::ostringstream item_path;
        item_path << path << '[' << i << ']';

        ExportedOutputSpec spec;
        if (parse_output_spec(values[i], spec, item_path.str(), error) != 0)
            return -1;
        specs.push_back(spec);
    }

    return 0;
}

static int parse_graph_signature(const JsonValue& value, std::vector<ExportedInputSpec>& input_specs, std::vector<ExportedOutputSpec>& output_specs, const std::string& path, ExportedSchemaError& error)
{
    if (value.type() != JSON_OBJECT)
        return schema_error(error, path, "expected object");

    const JsonValue* inputs = required_field(value, "input_specs", path, error);
    if (!inputs)
        return -1;
    if (parse_input_specs(*inputs, input_specs, path + ".input_specs", error) != 0)
        return -1;

    const JsonValue* outputs = required_field(value, "output_specs", path, error);
    if (!outputs)
        return -1;
    return parse_output_specs(*outputs, output_specs, path + ".output_specs", error);
}

static int parse_output_tree_node(const JsonValue& value, ExportedTreeSpec& tree_spec, const std::string& path, ExportedSchemaError& error)
{
    if (value.type() != JSON_OBJECT)
        return schema_error(error, path, "output treespec node must be an object");

    const JsonValue* type = required_field(value, "type", path, error);
    if (!type)
        return -1;
    const JsonValue* context = required_field(value, "context", path, error);
    if (!context)
        return -1;
    const JsonValue* children_spec = required_field(value, "children_spec", path, error);
    if (!children_spec)
        return -1;
    if (children_spec->type() != JSON_ARRAY)
        return schema_error(error, path, "output treespec children_spec must be an array");

    if (type->type() == JSON_NULL)
    {
        if (context->type() != JSON_NULL)
            return schema_error(error, path, "leaf context must be null");
        if (!children_spec->as_array().empty())
            return schema_error(error, path, "leaf must not have children");

        tree_spec.type = EXPORTED_TREE_SPEC_LEAF;
        return 0;
    }

    if (type->type() != JSON_STRING)
        return schema_error(error, path, "output treespec type must be a string or null");

    const std::string& type_name = type->as_string();
    if (type_name == "builtins.tuple")
        tree_spec.type = EXPORTED_TREE_SPEC_TUPLE;
    else if (type_name == "builtins.list")
        tree_spec.type = EXPORTED_TREE_SPEC_LIST;
    else
        return schema_error(error, path, "unsupported output treespec type " + type_name);

    if (context->type() != JSON_STRING || context->as_string() != "null")
        return schema_error(error, path, "container context must be the string null");

    const std::vector<JsonValue>& children = children_spec->as_array();
    tree_spec.children.reserve(children.size());
    for (size_t i = 0; i < children.size(); i++)
    {
        ExportedTreeSpec child;
        if (parse_output_tree_node(children[i], child, path, error) != 0)
            return -1;
        tree_spec.children.push_back(child);
    }

    return 0;
}

static int parse_serialized_output_tree(const std::string& serialized, ExportedTreeSpec& tree_spec, const std::string& path, ExportedSchemaError& error)
{
    JsonValue value;
    JsonParseError parse_error;
    JsonParseOptions options;
    if (parse_json(serialized.data(), serialized.size(), value, parse_error, options) != 0)
    {
        std::ostringstream message;
        message << "invalid output treespec JSON at byte " << parse_error.byte_offset << ": " << parse_error.message;
        return schema_error(error, path, message.str());
    }

    if (value.type() != JSON_ARRAY || value.as_array().size() != 2)
        return schema_error(error, path, "output treespec must be a protocol and tree pair");

    const JsonValue& protocol = value.as_array()[0];
    if (protocol.type() != JSON_INT64)
        return schema_error(error, path, "output treespec protocol must be an integer");
    if (protocol.as_int64() != 1)
    {
        std::ostringstream message;
        message << "unsupported output treespec protocol " << protocol.as_int64();
        return schema_error(error, path, message.str());
    }

    return parse_output_tree_node(value.as_array()[1], tree_spec, path, error);
}

static int parse_module_call_graph(const JsonValue& value, ExportedTreeSpec& output_tree_spec, std::string& output_tree_path, const std::string& path, ExportedSchemaError& error)
{
    if (value.type() != JSON_ARRAY)
        return schema_error(error, path, "expected array");

    bool found_root = false;
    const std::vector<JsonValue>& entries = value.as_array();
    for (size_t i = 0; i < entries.size(); i++)
    {
        std::ostringstream entry_path_stream;
        entry_path_stream << path << '[' << i << ']';
        const std::string entry_path = entry_path_stream.str();

        if (entries[i].type() != JSON_OBJECT)
            return schema_error(error, entry_path, "expected object");

        const JsonValue* fqn = required_field(entries[i], "fqn", entry_path, error);
        if (!fqn)
            return -1;
        if (fqn->type() != JSON_STRING)
            return schema_error(error, entry_path + ".fqn", "expected string");
        if (!fqn->as_string().empty())
            continue;

        if (found_root)
            return schema_error(error, entry_path + ".fqn", "duplicate root module entry");
        found_root = true;

        const JsonValue* signature = required_field(entries[i], "signature", entry_path, error);
        if (!signature)
            return -1;
        if (signature->type() == JSON_NULL)
            return schema_error(error, entry_path + ".signature", "root module signature is required");
        if (signature->type() != JSON_OBJECT)
            return schema_error(error, entry_path + ".signature", "expected object");

        const JsonValue* out_spec = required_field(*signature, "out_spec", entry_path + ".signature", error);
        if (!out_spec)
            return -1;
        std::string serialized;
        output_tree_path = entry_path + ".signature.out_spec";
        if (read_string(*out_spec, serialized, output_tree_path, error) != 0)
            return -1;
        if (parse_serialized_output_tree(serialized, output_tree_spec, output_tree_path, error) != 0)
            return -1;
    }

    return 0;
}

static size_t output_tree_leaf_count(const ExportedTreeSpec& tree_spec)
{
    if (tree_spec.type == EXPORTED_TREE_SPEC_LEAF)
        return 1;

    size_t count = 0;
    for (size_t i = 0; i < tree_spec.children.size(); i++)
        count += output_tree_leaf_count(tree_spec.children[i]);
    return count;
}

static int validate_signature_tensor_arguments(const ExportedProgram& program, ExportedSchemaError& error)
{
    const std::string tensor_values_path = "$.graph_module.graph.tensor_values";
    for (size_t i = 0; i < program.input_specs.size(); i++)
    {
        if (validate_tensor_argument(program.input_specs[i].arg, program.graph.tensor_values, tensor_values_path, error) != 0)
            return -1;
    }
    for (size_t i = 0; i < program.output_specs.size(); i++)
    {
        if (validate_tensor_argument(program.output_specs[i].arg, program.graph.tensor_values, tensor_values_path, error) != 0)
            return -1;
    }

    return 0;
}

static int parse_range_constraints(const JsonValue& value, std::map<std::string, ExportedRangeConstraint>& constraints, const std::string& path, ExportedSchemaError& error)
{
    if (value.type() != JSON_OBJECT)
        return schema_error(error, path, "expected object");

    const std::map<std::string, JsonValue>& entries = value.as_object();
    for (std::map<std::string, JsonValue>::const_iterator it = entries.begin(); it != entries.end(); ++it)
    {
        const std::string item_path = schema_map_key_path(path, it->first);
        if (it->first.empty())
            return schema_error(error, item_path, "range constraint name must not be empty");
        if (it->second.type() != JSON_OBJECT)
            return schema_error(error, item_path, "expected object");

        ExportedRangeConstraint constraint;
        const JsonValue* min_val = required_field(it->second, "min_val", item_path, error);
        if (!min_val)
            return -1;
        if (min_val->type() != JSON_NULL)
        {
            if (read_integer(*min_val, constraint.min, item_path + ".min_val", error) != 0)
                return -1;
            constraint.has_min = true;
        }

        const JsonValue* max_val = required_field(it->second, "max_val", item_path, error);
        if (!max_val)
            return -1;
        if (max_val->type() != JSON_NULL)
        {
            if (read_integer(*max_val, constraint.max, item_path + ".max_val", error) != 0)
                return -1;
            constraint.has_max = true;
        }

        if (constraint.has_min && constraint.has_max && constraint.min > constraint.max)
            return schema_error(error, item_path, "range constraint minimum exceeds maximum");

        constraints[it->first] = constraint;
    }

    return 0;
}

int parse_exported_program(const JsonValue& value, ExportedProgram& program, ExportedSchemaError& error)
{
    program = ExportedProgram();
    clear_schema_error(error);

    if (value.type() != JSON_OBJECT)
        return schema_error(error, "$", "expected object");

    ExportedProgram parsed_program;
    if (parse_exported_program_header(value, parsed_program.header, error) != 0)
        return -1;

    const JsonValue* graph_module = required_field(value, "graph_module", "$", error);
    if (!graph_module)
        return -1;
    if (graph_module->type() != JSON_OBJECT)
        return schema_error(error, "$.graph_module", "expected object");

    const JsonValue* graph = required_field(*graph_module, "graph", "$.graph_module", error);
    if (!graph)
        return -1;
    if (parse_exported_graph(*graph, parsed_program.graph, "$.graph_module.graph", error) != 0)
        return -1;

    const JsonValue* signature = required_field(*graph_module, "signature", "$.graph_module", error);
    if (!signature)
        return -1;
    if (parse_graph_signature(*signature, parsed_program.input_specs, parsed_program.output_specs, "$.graph_module.signature", error) != 0)
        return -1;

    const JsonValue* module_call_graph = required_field(*graph_module, "module_call_graph", "$.graph_module", error);
    if (!module_call_graph)
        return -1;
    std::string output_tree_path;
    if (parse_module_call_graph(*module_call_graph, parsed_program.output_tree_spec, output_tree_path, "$.graph_module.module_call_graph", error) != 0)
        return -1;

    const JsonValue* range_constraints = required_field(value, "range_constraints", "$", error);
    if (!range_constraints)
        return -1;
    if (range_constraints->type() != JSON_OBJECT)
        return schema_error(error, "$.range_constraints", "expected object");
    if (parse_range_constraints(*range_constraints, parsed_program.range_constraints, "$.range_constraints", error) != 0)
        return -1;

    if (parsed_program.input_specs.size() != parsed_program.graph.inputs.size())
        return schema_error(error, "$.graph_module.signature.input_specs", "input spec count does not match graph inputs");
    if (parsed_program.output_specs.size() != parsed_program.graph.outputs.size())
        return schema_error(error, "$.graph_module.signature.output_specs", "output spec count does not match graph outputs");
    if (parsed_program.output_tree_spec.type != EXPORTED_TREE_SPEC_NONE && output_tree_leaf_count(parsed_program.output_tree_spec) != parsed_program.graph.outputs.size())
        return schema_error(error, output_tree_path, "output treespec leaf count does not match graph outputs");
    if (validate_signature_tensor_arguments(parsed_program, error) != 0)
        return -1;

    program = parsed_program;
    return 0;
}

} // namespace pnnx
