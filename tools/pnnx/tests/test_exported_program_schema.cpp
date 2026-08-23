// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "exported_program_schema.h"
#include "json_reader.h"
#include "pt2_archive.h"

#include <stdio.h>

#include <sstream>
#include <string>
#include <vector>

static int test_failures = 0;
static int test_paths = 0;

static void check(bool condition, const std::string& label, const std::string& message)
{
    if (condition)
        return;

    fprintf(stderr, "schema test failed %s %s\n", label.c_str(), message.c_str());
    test_failures++;
}

static bool parse_document(const std::string& text, pnnx::JsonValue& value, const std::string& label)
{
    pnnx::JsonParseError error;
    pnnx::JsonParseOptions options;
    if (pnnx::parse_json(text.data(), text.size(), value, error, options) == 0)
        return true;

    std::ostringstream message;
    message << "fixture json failed at byte " << error.byte_offset << ": " << error.message;
    check(false, label, message.str());
    return false;
}

static std::string header_json(int schema_major, int schema_minor, const std::string& torch_version, const std::string& opset = "\"aten\":10")
{
    std::ostringstream text;
    text << "{\"schema_version\":{\"major\":" << schema_major
         << ",\"minor\":" << schema_minor
         << "},\"torch_version\":\"" << torch_version
         << "\",\"opset_version\":{" << opset << "}}";
    return text.str();
}

static std::string tensor_meta_json(const std::string& sizes = "[{\"as_int\":3},{\"as_int\":4}]",
                                    const std::string& strides = "[{\"as_int\":4},{\"as_int\":1}]",
                                    const std::string& storage_offset = "{\"as_int\":0}",
                                    const std::string& device_index = "null")
{
    return "{\"dtype\":7,\"sizes\":" + sizes
           + ",\"requires_grad\":true,\"device\":{\"type\":\"cpu\",\"index\":" + device_index
           + "},\"strides\":" + strides
           + ",\"storage_offset\":" + storage_offset
           + ",\"layout\":7}";
}

static void expect_header(const std::string& text, int schema_major, int schema_minor, const std::string& torch_version, int64_t aten_opset, const std::string& label)
{
    test_paths++;

    pnnx::JsonValue value;
    if (!parse_document(text, value, label))
        return;

    pnnx::ExportedProgramHeader header;
    pnnx::ExportedSchemaError error;
    error.path = "stale";
    error.message = "stale";
    if (pnnx::parse_exported_program_header(value, header, error) != 0)
    {
        check(false, label, error.path + " " + error.message);
        return;
    }

    check(error.path.empty() && error.message.empty(), label, "success did not clear error");
    check(header.schema_major == schema_major, label, "unexpected schema major");
    check(header.schema_minor == schema_minor, label, "unexpected schema minor");
    check(header.torch_version == torch_version, label, "unexpected torch version");
    check(header.opset_version.count("aten") == 1 && header.opset_version.find("aten")->second == aten_opset, label, "unexpected aten opset");
}

static void expect_header_error(const std::string& text, const std::string& path, const std::string& message, const std::string& label)
{
    test_paths++;

    pnnx::JsonValue value;
    if (!parse_document(text, value, label))
        return;

    pnnx::ExportedProgramHeader header;
    header.schema_major = 99;
    header.schema_minor = 99;
    header.torch_version = "stale";
    header.opset_version["stale"] = 99;
    pnnx::ExportedSchemaError error;
    if (pnnx::parse_exported_program_header(value, header, error) == 0)
    {
        check(false, label, "unexpected success");
        return;
    }

    check(error.path == path, label, "unexpected path " + error.path);
    check(error.message.find(message) != std::string::npos, label, "unexpected message " + error.message);
    check(header.schema_major == 0 && header.schema_minor == 0 && header.torch_version.empty() && header.opset_version.empty(), label, "failure did not reset output");
}

static void test_headers()
{
    expect_header(header_json(8, 14, "2.9.0"), 8, 14, "2.9.0", 10, "header torch 2.9");
    expect_header(header_json(8, 15, "2.10.1+cpu"), 8, 15, "2.10.1+cpu", 10, "header torch 2.10");
    expect_header(header_json(8, 17, "2.11.0"), 8, 17, "2.11.0", 10, "header torch 2.11");
    expect_header(header_json(8, 20, "2.12.1+cu126", "\"aten\":10,\"custom\":3"), 8, 20, "2.12.1+cu126", 10, "header torch 2.12");
    expect_header(header_json(8, 20, "2.12.0a0+gitabcdef"), 8, 20, "2.12.0a0+gitabcdef", 10, "header nightly suffix");

    expect_header_error("[]", "$", "expected object", "header root type");
    expect_header_error("{\"torch_version\":\"2.12.1\",\"opset_version\":{\"aten\":10}}", "$.schema_version", "missing required field", "header missing schema");
    expect_header_error("{\"schema_version\":[],\"torch_version\":\"2.12.1\",\"opset_version\":{\"aten\":10}}", "$.schema_version", "expected object", "header schema type");
    expect_header_error("{\"schema_version\":{\"minor\":20},\"torch_version\":\"2.12.1\",\"opset_version\":{\"aten\":10}}", "$.schema_version.major", "missing required field", "header missing major");
    expect_header_error("{\"schema_version\":{\"major\":8},\"torch_version\":\"2.12.1\",\"opset_version\":{\"aten\":10}}", "$.schema_version.minor", "missing required field", "header missing minor");
    expect_header_error("{\"schema_version\":{\"major\":8,\"minor\":20},\"opset_version\":{\"aten\":10}}", "$.torch_version", "missing required field", "header missing torch version");
    expect_header_error(header_json(8, 8, "2.8.0"), "$.torch_version", "legacy pickled-payload", "header legacy 2.8");
    expect_header_error(header_json(8, 7, "2.7.1"), "$.torch_version", "legacy exported program producer", "header legacy 2.7");
    expect_header_error(header_json(8, 21, "2.13.0"), "$.torch_version", "untested torch producer", "header future producer");
    expect_header_error(header_json(9, 1, "2.12.1"), "$.schema_version.major", "incompatible schema major", "header schema major");
    expect_header_error(header_json(8, 17, "2.12.1"), "$.schema_version.minor", "does not match torch producer", "header schema producer mismatch");
    expect_header_error(header_json(8, 20, "v2.12.1"), "$.torch_version", "invalid torch producer version", "header version prefix");
    expect_header_error(header_json(8, 20, "2.12x"), "$.torch_version", "invalid torch producer version", "header version suffix");
    expect_header_error("{\"schema_version\":{\"major\":8,\"minor\":20},\"torch_version\":12,\"opset_version\":{\"aten\":10}}", "$.torch_version", "expected string", "header version type");
    expect_header_error("{\"schema_version\":{\"major\":8,\"minor\":20},\"torch_version\":\"2.12.1\"}", "$.opset_version", "missing required field", "header missing opset");
    expect_header_error("{\"schema_version\":{\"major\":8,\"minor\":20},\"torch_version\":\"2.12.1\",\"opset_version\":[]}", "$.opset_version", "expected object", "header opset type");
    expect_header_error(header_json(8, 20, "2.12.1", "\"custom\":3"), "$.opset_version.aten", "missing required field", "header missing aten");
    expect_header_error(header_json(8, 20, "2.12.1", "\"aten\":\"10\""), "$.opset_version.aten", "expected integer", "header aten type");
    expect_header_error(header_json(8, 20, "2.12.1", "\"aten\":10,\"custom\":true"), "$.opset_version.custom", "expected integer", "header custom opset type");
    expect_header_error("{\"schema_version\":{\"major\":\"8\",\"minor\":20},\"torch_version\":\"2.12.1\",\"opset_version\":{\"aten\":10}}", "$.schema_version.major", "expected integer", "header schema major type");
    expect_header_error("{\"schema_version\":{\"major\":8,\"minor\":-1},\"torch_version\":\"2.12.1\",\"opset_version\":{\"aten\":10}}", "$.schema_version.minor", "schema minor must be non-negative", "header negative schema minor");
    expect_header_error(header_json(8, 20, "2.12.1", "\"aten\":-1"), "$.opset_version.aten", "opset version must be non-negative", "header negative aten opset");
}

static void expect_tensor(const std::string& text, const std::string& label)
{
    test_paths++;

    pnnx::JsonValue value;
    if (!parse_document(text, value, label))
        return;

    pnnx::ExportedTensorMeta meta;
    pnnx::ExportedSchemaError error;
    error.path = "stale";
    error.message = "stale";
    if (pnnx::parse_exported_tensor_meta(value, meta, error, "$.tensor_meta") != 0)
    {
        check(false, label, error.path + " " + error.message);
        return;
    }

    check(error.path.empty() && error.message.empty(), label, "success did not clear error");
    check(meta.dtype == 7, label, "unexpected dtype");
    check(meta.sizes.size() == 2 && meta.sizes[0] == 3 && meta.sizes[1] == 4, label, "unexpected sizes");
    check(meta.strides.size() == 2 && meta.strides[0] == 4 && meta.strides[1] == 1, label, "unexpected strides");
    check(meta.storage_offset == 0, label, "unexpected storage offset");
    check(meta.layout == 7, label, "unexpected layout");
    check(meta.requires_grad, label, "unexpected requires_grad");
    check(meta.device_type == "cpu", label, "unexpected device type");
    check(!meta.has_device_index, label, "unexpected device index");
}

static void expect_tensor_error(const std::string& text, const std::string& path, const std::string& message, const std::string& label)
{
    test_paths++;

    pnnx::JsonValue value;
    if (!parse_document(text, value, label))
        return;

    pnnx::ExportedTensorMeta meta;
    meta.dtype = 99;
    meta.sizes.push_back(99);
    meta.strides.push_back(99);
    meta.storage_offset = 99;
    meta.layout = 99;
    meta.requires_grad = true;
    meta.device_type = "stale";
    meta.device_index = 99;
    meta.has_device_index = true;
    pnnx::ExportedSchemaError error;
    if (pnnx::parse_exported_tensor_meta(value, meta, error, "$.tensor_meta") == 0)
    {
        check(false, label, "unexpected success");
        return;
    }

    check(error.path == path, label, "unexpected path " + error.path);
    check(error.message.find(message) != std::string::npos, label, "unexpected message " + error.message);
    check(meta.dtype == 0 && meta.sizes.empty() && meta.strides.empty() && meta.storage_offset == 0 && meta.layout == 0 && !meta.requires_grad && meta.device_type.empty() && meta.device_index == 0 && !meta.has_device_index, label, "failure did not reset output");
}

static void test_tensor_meta()
{
    expect_tensor(tensor_meta_json(), "tensor real static");

    test_paths++;
    {
        const std::string label = "tensor cuda index";
        pnnx::JsonValue value;
        const std::string text = "{\"dtype\":6,\"sizes\":[],\"requires_grad\":false,\"device\":{\"type\":\"cuda\",\"index\":2},\"strides\":[],\"storage_offset\":{\"as_int\":0},\"layout\":7}";
        if (parse_document(text, value, label))
        {
            pnnx::ExportedTensorMeta meta;
            pnnx::ExportedSchemaError error;
            const int result = pnnx::parse_exported_tensor_meta(value, meta, error, "$.tensor_meta");
            check(result == 0, label, error.path + " " + error.message);
            if (result == 0)
            {
                check(meta.sizes.empty() && meta.strides.empty(), label, "scalar rank changed");
                check(meta.device_type == "cuda" && meta.has_device_index && meta.device_index == 2, label, "device index lost");
            }
        }
    }

    expect_tensor_error("[]", "$.tensor_meta", "expected object", "tensor root type");
    expect_tensor_error("{\"sizes\":[],\"requires_grad\":false,\"device\":{\"type\":\"cpu\",\"index\":null},\"strides\":[],\"storage_offset\":{\"as_int\":0},\"layout\":7}", "$.tensor_meta.dtype", "missing required field", "tensor missing dtype");
    expect_tensor_error("{\"dtype\":7,\"requires_grad\":false,\"device\":{\"type\":\"cpu\",\"index\":null},\"strides\":[],\"storage_offset\":{\"as_int\":0},\"layout\":7}", "$.tensor_meta.sizes", "missing required field", "tensor missing sizes");
    expect_tensor_error("{\"dtype\":7,\"sizes\":[],\"device\":{\"type\":\"cpu\",\"index\":null},\"strides\":[],\"storage_offset\":{\"as_int\":0},\"layout\":7}", "$.tensor_meta.requires_grad", "missing required field", "tensor missing requires grad");
    expect_tensor_error("{\"dtype\":7,\"sizes\":[],\"requires_grad\":false,\"strides\":[],\"storage_offset\":{\"as_int\":0},\"layout\":7}", "$.tensor_meta.device", "missing required field", "tensor missing device");
    expect_tensor_error("{\"dtype\":7,\"sizes\":[],\"requires_grad\":false,\"device\":{\"type\":\"cpu\",\"index\":null},\"storage_offset\":{\"as_int\":0},\"layout\":7}", "$.tensor_meta.strides", "missing required field", "tensor missing strides");
    expect_tensor_error("{\"dtype\":7,\"sizes\":[],\"requires_grad\":false,\"device\":{\"type\":\"cpu\",\"index\":null},\"strides\":[],\"layout\":7}", "$.tensor_meta.storage_offset", "missing required field", "tensor missing offset");
    expect_tensor_error("{\"dtype\":7,\"sizes\":[],\"requires_grad\":false,\"device\":{\"type\":\"cpu\",\"index\":null},\"strides\":[],\"storage_offset\":{\"as_int\":0}}", "$.tensor_meta.layout", "missing required field", "tensor missing layout");
    expect_tensor_error(tensor_meta_json("[{\"as_expr\":{\"expr_str\":\"s0\",\"hint\":{\"as_int\":3}}},{\"as_int\":4}]"), "$.tensor_meta.sizes[0].as_expr", "dynamic tensor metadata is unsupported", "tensor symbolic size");
    expect_tensor_error(tensor_meta_json("[{\"as_name\":\"s0\"},{\"as_int\":4}]"), "$.tensor_meta.sizes[0].as_name", "dynamic tensor metadata is unsupported", "tensor named size");
    expect_tensor_error(tensor_meta_json("[{}, {\"as_int\":4}]"), "$.tensor_meta.sizes[0]", "expected static SymInt as_int", "tensor empty symint");
    expect_tensor_error(tensor_meta_json("[{\"as_int\":\"3\"},{\"as_int\":4}]"), "$.tensor_meta.sizes[0].as_int", "expected integer", "tensor size type");
    expect_tensor_error(tensor_meta_json("[{\"as_int\":-1},{\"as_int\":4}]"), "$.tensor_meta.sizes[0].as_int", "tensor size must be non-negative", "tensor negative size");
    expect_tensor_error(tensor_meta_json("[{\"as_int\":3},{\"as_int\":4}]", "[{\"as_int\":4}]"), "$.tensor_meta.strides", "rank does not match sizes", "tensor rank mismatch");
    expect_tensor_error(tensor_meta_json("[{\"as_int\":3},{\"as_int\":4}]", "[{\"as_expr\":{\"expr_str\":\"s0\"}},{\"as_int\":1}]"), "$.tensor_meta.strides[0].as_expr", "dynamic tensor metadata is unsupported", "tensor symbolic stride");
    expect_tensor_error(tensor_meta_json("[{\"as_int\":3},{\"as_int\":4}]", "[{\"as_int\":4},{\"as_int\":1}]", "{\"as_expr\":{\"expr_str\":\"s0\"}}"), "$.tensor_meta.storage_offset.as_expr", "dynamic tensor metadata is unsupported", "tensor symbolic offset");
    expect_tensor_error(tensor_meta_json("[{\"as_int\":3},{\"as_int\":4}]", "[{\"as_int\":4},{\"as_int\":1}]", "{\"as_int\":-1}"), "$.tensor_meta.storage_offset.as_int", "storage offset must be non-negative", "tensor negative offset");
    expect_tensor_error("{\"dtype\":7,\"sizes\":[],\"requires_grad\":false,\"device\":{\"index\":null},\"strides\":[],\"storage_offset\":{\"as_int\":0},\"layout\":7}", "$.tensor_meta.device.type", "missing required field", "tensor missing device type");
    expect_tensor_error("{\"dtype\":7,\"sizes\":[],\"requires_grad\":false,\"device\":{\"type\":\"cpu\"},\"strides\":[],\"storage_offset\":{\"as_int\":0},\"layout\":7}", "$.tensor_meta.device.index", "missing required field", "tensor missing device index");
    expect_tensor_error("{\"dtype\":7,\"sizes\":[],\"requires_grad\":false,\"device\":{\"type\":\"\",\"index\":null},\"strides\":[],\"storage_offset\":{\"as_int\":0},\"layout\":7}", "$.tensor_meta.device.type", "device type must not be empty", "tensor empty device type");
    expect_tensor_error(tensor_meta_json("[{\"as_int\":3},{\"as_int\":4}]", "[{\"as_int\":4},{\"as_int\":1}]", "{\"as_int\":0}", "-1"), "$.tensor_meta.device.index", "device index must be non-negative", "tensor negative device index");
    expect_tensor_error("{\"dtype\":\"float\",\"sizes\":[],\"requires_grad\":false,\"device\":{\"type\":\"cpu\",\"index\":null},\"strides\":[],\"storage_offset\":{\"as_int\":0},\"layout\":7}", "$.tensor_meta.dtype", "expected integer", "tensor dtype type");
    expect_tensor_error("{\"dtype\":7,\"sizes\":[],\"requires_grad\":0,\"device\":{\"type\":\"cpu\",\"index\":null},\"strides\":[],\"storage_offset\":{\"as_int\":0},\"layout\":7}", "$.tensor_meta.requires_grad", "expected boolean", "tensor requires grad type");
    expect_tensor_error("{\"dtype\":7,\"sizes\":[],\"requires_grad\":false,\"device\":[],\"strides\":[],\"storage_offset\":{\"as_int\":0},\"layout\":7}", "$.tensor_meta.device", "expected object", "tensor device type");
    expect_tensor_error(tensor_meta_json("[{\"as_int\":3},{\"as_int\":4}]", "[{\"as_int\":-4},{\"as_int\":1}]"), "$.tensor_meta.strides[0].as_int", "tensor stride must be non-negative", "tensor negative stride");
    expect_tensor_error("{\"dtype\":7,\"sizes\":[],\"requires_grad\":false,\"device\":{\"type\":\"cpu\",\"index\":null},\"strides\":[],\"storage_offset\":{\"as_int\":0},\"layout\":-1}", "$.tensor_meta.layout", "tensor layout must be non-negative", "tensor negative layout");
}

static void expect_payload_error(const std::string& text, const std::string& path, const std::string& message, const std::string& label)
{
    test_paths++;

    pnnx::JsonValue value;
    if (!parse_document(text, value, label))
        return;

    pnnx::ExportedPayloadConfig config;
    config.entries["stale"].path_name = "stale";
    pnnx::ExportedSchemaError error;
    if (pnnx::parse_exported_payload_config(value, config, error) == 0)
    {
        check(false, label, "unexpected success");
        return;
    }

    check(error.path == path, label, "unexpected path " + error.path);
    check(error.message.find(message) != std::string::npos, label, "unexpected message " + error.message);
    check(config.entries.empty(), label, "failure did not reset output");
}

static void test_payload_config()
{
    test_paths++;
    {
        const std::string label = "payload real weights";
        const std::string text = "{\"config\":{\"linear.weight\":{\"path_name\":\"weight_0\",\"is_param\":true,\"use_pickle\":false,\"tensor_meta\":" + tensor_meta_json() + "},\"linear.bias\":{\"path_name\":\"weight_1\",\"is_param\":true,\"use_pickle\":false,\"tensor_meta\":" + tensor_meta_json("[{\"as_int\":3}]", "[{\"as_int\":1}]") + "}}}";
        pnnx::JsonValue value;
        if (parse_document(text, value, label))
        {
            pnnx::ExportedPayloadConfig config;
            pnnx::ExportedSchemaError error;
            const int result = pnnx::parse_exported_payload_config(value, config, error);
            check(result == 0, label, error.path + " " + error.message);
            if (result == 0)
            {
                check(config.entries.size() == 2, label, "unexpected entry count");
                const pnnx::ExportedPayloadEntry& weight = config.entries.find("linear.weight")->second;
                const pnnx::ExportedPayloadEntry& bias = config.entries.find("linear.bias")->second;
                check(weight.path_name == "weight_0" && weight.is_param && !weight.use_pickle && weight.has_tensor_meta, label, "weight flags changed");
                check(weight.tensor_meta.sizes.size() == 2 && weight.tensor_meta.sizes[0] == 3 && weight.tensor_meta.sizes[1] == 4, label, "weight shape changed");
                check(bias.path_name == "weight_1" && bias.tensor_meta.sizes.size() == 1 && bias.tensor_meta.sizes[0] == 3, label, "bias metadata changed");
            }
        }
    }

    test_paths++;
    {
        const std::string label = "payload empty constants";
        pnnx::JsonValue value;
        if (parse_document("{\"config\":{}}", value, label))
        {
            pnnx::ExportedPayloadConfig config;
            pnnx::ExportedSchemaError error;
            const int result = pnnx::parse_exported_payload_config(value, config, error);
            check(result == 0, label, error.path + " " + error.message);
            check(config.entries.empty(), label, "empty config gained entries");
        }
    }

    test_paths++;
    {
        const std::string label = "payload pickled custom object";
        const std::string text = "{\"config\":{\"custom.value\":{\"path_name\":\"custom_obj_0\",\"is_param\":false,\"use_pickle\":true,\"tensor_meta\":null}}}";
        pnnx::JsonValue value;
        if (parse_document(text, value, label))
        {
            pnnx::ExportedPayloadConfig config;
            pnnx::ExportedSchemaError error;
            const int result = pnnx::parse_exported_payload_config(value, config, error);
            check(result == 0, label, error.path + " " + error.message);
            if (result == 0)
            {
                const pnnx::ExportedPayloadEntry& entry = config.entries.find("custom.value")->second;
                check(entry.use_pickle && !entry.has_tensor_meta, label, "optional tensor metadata changed");
            }
        }
    }

    test_paths++;
    {
        const std::string label = "payload pickled tensor metadata";
        const std::string text = "{\"config\":{\"subclass.value\":{\"path_name\":\"custom_obj_0\",\"is_param\":false,\"use_pickle\":true,\"tensor_meta\":" + tensor_meta_json() + "}}}";
        pnnx::JsonValue value;
        if (parse_document(text, value, label))
        {
            pnnx::ExportedPayloadConfig config;
            pnnx::ExportedSchemaError error;
            const int result = pnnx::parse_exported_payload_config(value, config, error);
            check(result == 0, label, error.path + " " + error.message);
            if (result == 0)
            {
                const pnnx::ExportedPayloadEntry& entry = config.entries.find("subclass.value")->second;
                check(entry.use_pickle && entry.has_tensor_meta, label, "pickled tensor metadata lost");
            }
        }
    }

    test_paths++;
    {
        const std::string label = "payload aliased path";
        const std::string meta = tensor_meta_json();
        const std::string text = "{\"config\":{\"first\":{\"path_name\":\"weight_0\",\"is_param\":true,\"use_pickle\":false,\"tensor_meta\":" + meta + "},\"second\":{\"path_name\":\"weight_0\",\"is_param\":true,\"use_pickle\":false,\"tensor_meta\":" + meta + "}}}";
        pnnx::JsonValue value;
        if (parse_document(text, value, label))
        {
            pnnx::ExportedPayloadConfig config;
            pnnx::ExportedSchemaError error;
            const int result = pnnx::parse_exported_payload_config(value, config, error);
            check(result == 0 && config.entries.size() == 2, label, error.path + " " + error.message);
        }
    }

    expect_payload_error("[]", "$", "expected object", "payload root type");
    expect_payload_error("{}", "$.config", "missing required field", "payload missing config");
    expect_payload_error("{\"config\":[]}", "$.config", "expected object", "payload config type");
    expect_payload_error("{\"config\":{\"\":{\"path_name\":\"weight_0\",\"is_param\":true,\"use_pickle\":false,\"tensor_meta\":" + tensor_meta_json() + "}}}", "$.config[\"\"]", "payload name must not be empty", "payload empty fqn");
    expect_payload_error("{\"config\":{\"linear.weight\":[]}}", "$.config[\"linear.weight\"]", "expected object", "payload entry type");
    expect_payload_error("{\"config\":{\"linear.weight\":{\"is_param\":true,\"use_pickle\":false,\"tensor_meta\":" + tensor_meta_json() + "}}}", "$.config[\"linear.weight\"].path_name", "missing required field", "payload missing path");
    expect_payload_error("{\"config\":{\"linear.weight\":{\"path_name\":\"../weight_0\",\"is_param\":true,\"use_pickle\":false,\"tensor_meta\":" + tensor_meta_json() + "}}}", "$.config[\"linear.weight\"].path_name", "single archive path component", "payload unsafe path");
    expect_payload_error("{\"config\":{\"linear.weight\":{\"path_name\":\"weight_0\",\"use_pickle\":false,\"tensor_meta\":" + tensor_meta_json() + "}}}", "$.config[\"linear.weight\"].is_param", "missing required field", "payload missing is param");
    expect_payload_error("{\"config\":{\"linear.weight\":{\"path_name\":\"weight_0\",\"is_param\":true,\"tensor_meta\":" + tensor_meta_json() + "}}}", "$.config[\"linear.weight\"].use_pickle", "missing required field", "payload missing pickle flag");
    expect_payload_error("{\"config\":{\"linear.weight\":{\"path_name\":\"weight_0\",\"is_param\":true,\"use_pickle\":false}}}", "$.config[\"linear.weight\"].tensor_meta", "missing required field", "payload missing tensor meta");
    expect_payload_error("{\"config\":{\"linear.weight\":{\"path_name\":\"weight_0\",\"is_param\":true,\"use_pickle\":false,\"tensor_meta\":null}}}", "$.config[\"linear.weight\"].tensor_meta", "non-pickled payload requires tensor metadata", "payload null tensor meta");
    expect_payload_error("{\"config\":{\"linear.weight\":{\"path_name\":\"weight_0\",\"is_param\":true,\"use_pickle\":false,\"tensor_meta\":{}}}}", "$.config[\"linear.weight\"].tensor_meta.dtype", "missing required field", "payload nested tensor error");
    expect_payload_error("{\"config\":{\"linear.weight\":{\"path_name\":7,\"is_param\":true,\"use_pickle\":false,\"tensor_meta\":" + tensor_meta_json() + "}}}", "$.config[\"linear.weight\"].path_name", "expected string", "payload path type");
    expect_payload_error("{\"config\":{\"linear.weight\":{\"path_name\":\"weight_0\",\"is_param\":1,\"use_pickle\":false,\"tensor_meta\":" + tensor_meta_json() + "}}}", "$.config[\"linear.weight\"].is_param", "expected boolean", "payload is param type");
    expect_payload_error("{\"config\":{\"linear.weight\":{\"path_name\":\"weight_0\",\"is_param\":true,\"use_pickle\":0,\"tensor_meta\":" + tensor_meta_json() + "}}}", "$.config[\"linear.weight\"].use_pickle", "expected boolean", "payload pickle type");
    expect_payload_error("{\"config\":{\"linear.weight\":{\"path_name\":\"weight_0\",\"is_param\":true,\"use_pickle\":false,\"tensor_meta\":true}}}", "$.config[\"linear.weight\"].tensor_meta", "expected object", "payload tensor meta type");
}

static std::string join_ints(const std::vector<int64_t>& values)
{
    std::ostringstream text;
    for (size_t i = 0; i < values.size(); i++)
    {
        if (i != 0)
            text << ',';
        text << values[i];
    }

    return text.str();
}

static void print_payload_config(const char* config_name, const pnnx::ExportedPayloadConfig& config)
{
    for (std::map<std::string, pnnx::ExportedPayloadEntry>::const_iterator it = config.entries.begin(); it != config.entries.end(); ++it)
    {
        const pnnx::ExportedPayloadEntry& entry = it->second;
        const pnnx::ExportedTensorMeta& meta = entry.tensor_meta;
        fprintf(stdout, "payload|%s|%s|%s|%d|%d|%d|%lld|%s|%s|%lld|%s|",
                config_name,
                it->first.c_str(),
                entry.path_name.c_str(),
                entry.is_param ? 1 : 0,
                entry.use_pickle ? 1 : 0,
                entry.has_tensor_meta ? 1 : 0,
                (long long)meta.dtype,
                join_ints(meta.sizes).c_str(),
                join_ints(meta.strides).c_str(),
                (long long)meta.storage_offset,
                meta.device_type.c_str());
        if (meta.has_device_index)
            fprintf(stdout, "%lld", (long long)meta.device_index);
        else
            fprintf(stdout, "null");
        fprintf(stdout, "|%lld|%d\n", (long long)meta.layout, meta.requires_grad ? 1 : 0);
    }
}

static int inspect_package(const std::string& path)
{
    pnnx::Pt2ArchiveReader reader;
    std::string archive_error;
    if (reader.open(path, archive_error) != 0)
    {
        fprintf(stderr, "%s\n", archive_error.c_str());
        return 1;
    }

    pnnx::JsonValue model_json;
    pnnx::JsonValue weights_json;
    pnnx::JsonValue constants_json;
    if (reader.read_json(reader.layout().model_json_path, model_json, archive_error) != 0
        || reader.read_json(reader.layout().weights_config_path, weights_json, archive_error) != 0
        || reader.read_json(reader.layout().constants_config_path, constants_json, archive_error) != 0)
    {
        fprintf(stderr, "%s\n", archive_error.c_str());
        return 1;
    }

    pnnx::ExportedProgramHeader header;
    pnnx::ExportedPayloadConfig weights;
    pnnx::ExportedPayloadConfig constants;
    pnnx::ExportedSchemaError schema_error;
    if (pnnx::parse_exported_program_header(model_json, header, schema_error) != 0
        || pnnx::parse_exported_payload_config(weights_json, weights, schema_error) != 0
        || pnnx::parse_exported_payload_config(constants_json, constants, schema_error) != 0)
    {
        fprintf(stderr, "%s %s\n", schema_error.path.c_str(), schema_error.message.c_str());
        return 1;
    }

    fprintf(stdout, "header|%d|%d|%s|%lld|%llu|%llu\n",
            header.schema_major,
            header.schema_minor,
            header.torch_version.c_str(),
            (long long)header.opset_version.find("aten")->second,
            (unsigned long long)weights.entries.size(),
            (unsigned long long)constants.entries.size());

    print_payload_config("weights", weights);
    print_payload_config("constants", constants);

    return 0;
}

int main(int argc, char** argv)
{
    if (argc == 3 && std::string(argv[1]) == "inspect")
        return inspect_package(argv[2]);
    if (argc != 1)
        return 2;

    test_headers();
    test_tensor_meta();
    test_payload_config();

    if (test_failures != 0)
    {
        fprintf(stderr, "%d exported program schema tests failed across %d paths\n", test_failures, test_paths);
        return 1;
    }

    fprintf(stdout, "exported program schema tests passed %d paths\n", test_paths);
    return 0;
}
