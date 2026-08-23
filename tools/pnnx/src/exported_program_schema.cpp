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

ExportedTensorMeta::ExportedTensorMeta()
{
    dtype = 0;
    storage_offset = 0;
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

    if (cursor.position == text.size())
        return true;

    if (text[cursor.position] == '.')
    {
        cursor.position++;
        int patch = 0;
        if (!parse_version_component(cursor, patch))
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
    if (torch_version_value.major != 2 || torch_version_value.minor > 12)
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

static int parse_static_symint(const JsonValue& value, int64_t& result, const std::string& path, ExportedSchemaError& error)
{
    if (value.type() != JSON_OBJECT)
        return schema_error(error, path, "expected static SymInt object");

    if (value.find("as_expr"))
        return schema_error(error, path + ".as_expr", "dynamic tensor metadata is unsupported");
    if (value.find("as_name"))
        return schema_error(error, path + ".as_name", "dynamic tensor metadata is unsupported");

    const JsonValue* as_int = value.find("as_int");
    if (!as_int)
        return schema_error(error, path, "expected static SymInt as_int");
    if (value.as_object().size() != 1)
        return schema_error(error, path, "static SymInt union must contain only as_int");

    return read_integer(*as_int, result, path + ".as_int", error);
}

static int parse_static_symint_array(const JsonValue& value, std::vector<int64_t>& result, const std::string& path, bool is_size, ExportedSchemaError& error)
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
        if (parse_static_symint(values[i], item, item_path.str(), error) != 0)
            return -1;
        if (item < 0)
            return schema_error(error, item_path.str() + ".as_int", is_size ? "tensor size must be non-negative" : "tensor stride must be non-negative");

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
    if (parse_static_symint_array(*sizes, parsed_meta.sizes, tensor_path + ".sizes", true, error) != 0)
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
    if (parse_static_symint_array(*strides, parsed_meta.strides, tensor_path + ".strides", false, error) != 0)
        return -1;
    if (parsed_meta.strides.size() != parsed_meta.sizes.size())
        return schema_error(error, tensor_path + ".strides", "tensor stride rank does not match sizes");

    const JsonValue* storage_offset = required_field(value, "storage_offset", tensor_path, error);
    if (!storage_offset)
        return -1;
    if (parse_static_symint(*storage_offset, parsed_meta.storage_offset, tensor_path + ".storage_offset", error) != 0)
        return -1;
    if (parsed_meta.storage_offset < 0)
        return schema_error(error, tensor_path + ".storage_offset.as_int", "storage offset must be non-negative");

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

} // namespace pnnx
