// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "exported_program_schema.h"
#include "json_reader.h"
#include "pt2_archive.h"

#include <stdio.h>

#include <cmath>
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

static std::string tensor_argument_json(const std::string& name)
{
    return "{\"as_tensor\":{\"name\":\"" + name + "\"}}";
}

static std::string graph_argument_json(const std::string& name)
{
    return "{\"as_graph\":{\"name\":\"" + name
           + "\",\"graph\":{\"inputs\":[" + tensor_argument_json("sub_x")
           + "],\"outputs\":[" + tensor_argument_json("sub_y")
           + "],\"nodes\":[{\"target\":\"torch.ops.aten.relu.default\",\"inputs\":[{\"name\":\"self\",\"arg\":" + tensor_argument_json("sub_x")
           + ",\"kind\":1}],\"outputs\":[" + tensor_argument_json("sub_y")
           + "],\"metadata\":{},\"is_hop_single_tensor_return\":null,\"name\":\"sub_y\"}],\"tensor_values\":{\"sub_x\":" + tensor_meta_json("[{\"as_int\":2}]", "[{\"as_int\":1}]")
           + ",\"sub_y\":" + tensor_meta_json("[{\"as_int\":2}]", "[{\"as_int\":1}]")
           + "},\"sym_int_values\":{},\"sym_bool_values\":{},\"is_single_tensor_return\":false,\"custom_obj_values\":{},\"sym_float_values\":{}}}}";
}

static std::string json_string(const std::string& value)
{
    std::string result = "\"";
    for (size_t i = 0; i < value.size(); i++)
    {
        const char ch = value[i];
        if (ch == '"' || ch == '\\')
            result.push_back('\\');
        result.push_back(ch);
    }
    result.push_back('"');
    return result;
}

static std::string module_call_graph_json(const std::string& out_spec)
{
    const std::string leaf = "[1,{\"type\":null,\"context\":null,\"children_spec\":[]}]";
    return "[{\"fqn\":\"\",\"signature\":{\"inputs\":[],\"outputs\":[],\"in_spec\":" + json_string(leaf)
           + ",\"out_spec\":" + json_string(out_spec) + ",\"forward_arg_names\":[\"x\"]}}]";
}

struct ProgramFixture
{
    ProgramFixture()
        : schema_minor("20"),
          torch_version("2.12.1+cu126"),
          graph_inputs("[" + tensor_argument_json("x") + "]"),
          graph_outputs("[" + tensor_argument_json("y") + "]"),
          nodes("[{\"target\":\"torch.ops.aten.relu.default\",\"inputs\":[{\"name\":\"self\",\"arg\":" + tensor_argument_json("x") + ",\"kind\":1}],\"outputs\":[" + tensor_argument_json("y") + "],\"metadata\":{},\"is_hop_single_tensor_return\":null,\"name\":\"y\"}]"),
          tensor_values("{\"x\":" + tensor_meta_json("[{\"as_int\":2}]", "[{\"as_int\":1}]") + ",\"y\":" + tensor_meta_json("[{\"as_int\":2}]", "[{\"as_int\":1}]") + "}"),
          sym_int_values("{}"),
          sym_bool_values("{}"),
          custom_obj_values("{}"),
          sym_float_values("{}"),
          is_single_tensor_return("false"),
          input_specs("[{\"user_input\":{\"arg\":" + tensor_argument_json("x") + "}}]"),
          output_specs("[{\"user_output\":{\"arg\":" + tensor_argument_json("y") + "}}]"),
          module_call_graph(module_call_graph_json("[1,{\"type\":null,\"context\":null,\"children_spec\":[]}]")),
          range_constraints("{}")
    {
    }

    std::string json() const
    {
        return "{\"schema_version\":{\"major\":8,\"minor\":" + schema_minor + "},\"torch_version\":\"" + torch_version + "\",\"opset_version\":{\"aten\":10},\"graph_module\":{\"graph\":{\"inputs\":" + graph_inputs
               + ",\"outputs\":" + graph_outputs
               + ",\"nodes\":" + nodes
               + ",\"tensor_values\":" + tensor_values
               + ",\"sym_int_values\":" + sym_int_values
               + ",\"sym_bool_values\":" + sym_bool_values
               + ",\"is_single_tensor_return\":" + is_single_tensor_return
               + ",\"custom_obj_values\":" + custom_obj_values
               + ",\"sym_float_values\":" + sym_float_values
               + "},\"signature\":{\"input_specs\":" + input_specs
               + ",\"output_specs\":" + output_specs
               + "},\"module_call_graph\":" + module_call_graph
               + "},\"range_constraints\":" + range_constraints
               + "}";
    }

    std::string schema_minor;
    std::string torch_version;
    std::string graph_inputs;
    std::string graph_outputs;
    std::string nodes;
    std::string tensor_values;
    std::string sym_int_values;
    std::string sym_bool_values;
    std::string custom_obj_values;
    std::string sym_float_values;
    std::string is_single_tensor_return;
    std::string input_specs;
    std::string output_specs;
    std::string module_call_graph;
    std::string range_constraints;
};

static bool parse_program_fixture(const ProgramFixture& fixture, pnnx::ExportedProgram& program, const std::string& label)
{
    test_paths++;

    pnnx::JsonValue value;
    if (!parse_document(fixture.json(), value, label))
        return false;

    pnnx::ExportedSchemaError error;
    error.path = "stale";
    error.message = "stale";
    if (pnnx::parse_exported_program(value, program, error) != 0)
    {
        check(false, label, error.path + " " + error.message);
        return false;
    }

    check(error.path.empty() && error.message.empty(), label, "success did not clear error");
    return true;
}

static void expect_program_error(const ProgramFixture& fixture, const std::string& path, const std::string& message, const std::string& label)
{
    test_paths++;

    pnnx::JsonValue value;
    if (!parse_document(fixture.json(), value, label))
        return;

    pnnx::ExportedProgram program;
    program.header.schema_major = 99;
    program.graph.inputs.push_back(pnnx::ExportedArgument());
    program.input_specs.push_back(pnnx::ExportedInputSpec());
    pnnx::ExportedSchemaError error;
    if (pnnx::parse_exported_program(value, program, error) == 0)
    {
        check(false, label, "unexpected success");
        return;
    }

    check(error.path == path, label, "unexpected path " + error.path);
    check(error.message.find(message) != std::string::npos, label, "unexpected message " + error.message);
    check(program.header.schema_major == 0 && program.graph.inputs.empty() && program.input_specs.empty(), label, "failure did not reset output");
}

static bool parse_argument_case(const std::string& argument_json, const std::string& kind_field, pnnx::ExportedArgument& argument, pnnx::ExportedArgumentKind& kind, const std::string& label)
{
    ProgramFixture fixture;
    fixture.nodes = "[{\"target\":\"torch.ops.aten.clone.default\",\"inputs\":[{\"name\":\"value\",\"arg\":" + argument_json + kind_field + "}],\"outputs\":[" + tensor_argument_json("y") + "],\"metadata\":{},\"is_hop_single_tensor_return\":null,\"name\":\"y\"}]";

    pnnx::ExportedProgram program;
    if (!parse_program_fixture(fixture, program, label))
        return false;

    check(program.graph.nodes.size() == 1 && program.graph.nodes[0].inputs.size() == 1, label, "argument node shape changed");
    if (program.graph.nodes.size() != 1 || program.graph.nodes[0].inputs.size() != 1)
        return false;

    argument = program.graph.nodes[0].inputs[0].arg;
    kind = program.graph.nodes[0].inputs[0].kind;
    return true;
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
    expect_header(header_json(8, 20, "2.13.0+cpu"), 8, 20, "2.13.0+cpu", 10, "header torch 2.13");
    expect_header(header_json(8, 20, "2.12.0a0+gitabcdef"), 8, 20, "2.12.0a0+gitabcdef", 10, "header nightly suffix");

    expect_header_error("[]", "$", "expected object", "header root type");
    expect_header_error("{\"torch_version\":\"2.12.1\",\"opset_version\":{\"aten\":10}}", "$.schema_version", "missing required field", "header missing schema");
    expect_header_error("{\"schema_version\":[],\"torch_version\":\"2.12.1\",\"opset_version\":{\"aten\":10}}", "$.schema_version", "expected object", "header schema type");
    expect_header_error("{\"schema_version\":{\"minor\":20},\"torch_version\":\"2.12.1\",\"opset_version\":{\"aten\":10}}", "$.schema_version.major", "missing required field", "header missing major");
    expect_header_error("{\"schema_version\":{\"major\":8},\"torch_version\":\"2.12.1\",\"opset_version\":{\"aten\":10}}", "$.schema_version.minor", "missing required field", "header missing minor");
    expect_header_error("{\"schema_version\":{\"major\":8,\"minor\":20},\"opset_version\":{\"aten\":10}}", "$.torch_version", "missing required field", "header missing torch version");
    expect_header_error(header_json(8, 8, "2.8.0"), "$.torch_version", "legacy pickled-payload", "header legacy 2.8");
    expect_header_error(header_json(8, 7, "2.7.1"), "$.torch_version", "legacy exported program producer", "header legacy 2.7");
    expect_header_error(header_json(8, 21, "2.14.0"), "$.torch_version", "untested torch producer", "header future producer");
    expect_header_error(header_json(8, 20, "2.12.2"), "$.torch_version", "untested torch producer", "header future patch");
    expect_header_error(header_json(8, 20, "2.13.1"), "$.torch_version", "untested torch producer", "header current future patch");
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
    check(meta.sizes.size() == 2 && meta.sizes[0].type == pnnx::EXPORTED_SYM_INT_STATIC && meta.sizes[0].value == 3 && meta.sizes[1].value == 4, label, "unexpected sizes");
    check(meta.strides.size() == 2 && meta.strides[0].type == pnnx::EXPORTED_SYM_INT_STATIC && meta.strides[0].value == 4 && meta.strides[1].value == 1, label, "unexpected strides");
    check(meta.storage_offset.type == pnnx::EXPORTED_SYM_INT_STATIC && meta.storage_offset.value == 0, label, "unexpected storage offset");
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
    check(meta.dtype == 0 && meta.sizes.empty() && meta.strides.empty() && meta.storage_offset.type == pnnx::EXPORTED_SYM_INT_STATIC && meta.storage_offset.value == 0 && meta.layout == 0 && !meta.requires_grad && meta.device_type.empty() && meta.device_index == 0 && !meta.has_device_index, label, "failure did not reset output");
}

static void expect_symbolic_tensor_meta()
{
    const std::string label = "tensor symbolic metadata";
    const std::string text = tensor_meta_json(
        "[{\"as_expr\":{\"expr_str\":\"Symbol('u0', integer=True)\",\"hint\":null}},{\"as_int\":4}]",
        "[{\"as_expr\":{\"expr_str\":\"Mul(Integer(4), Symbol('u0', integer=True))\",\"hint\":{\"as_int\":12}}},{\"as_int\":1}]",
        "{\"as_expr\":{\"expr_str\":\"Symbol('u1', integer=True)\",\"hint\":{\"as_int\":2}}}");

    test_paths++;
    pnnx::JsonValue value;
    if (!parse_document(text, value, label))
        return;

    pnnx::ExportedTensorMeta meta;
    pnnx::ExportedSchemaError error;
    const int result = pnnx::parse_exported_tensor_meta(value, meta, error, "$.tensor_meta");
    check(result == 0, label, error.path + " " + error.message);
    if (result != 0)
        return;

    check(meta.sizes[0].type == pnnx::EXPORTED_SYM_INT_EXPRESSION && meta.sizes[0].expression == "Symbol('u0', integer=True)" && !meta.sizes[0].has_hint, label, "symbolic size changed");
    check(meta.strides[0].type == pnnx::EXPORTED_SYM_INT_EXPRESSION && meta.strides[0].expression == "Mul(Integer(4), Symbol('u0', integer=True))" && meta.strides[0].has_hint && meta.strides[0].hint == 12, label, "symbolic stride changed");
    check(meta.storage_offset.type == pnnx::EXPORTED_SYM_INT_EXPRESSION && meta.storage_offset.expression == "Symbol('u1', integer=True)" && meta.storage_offset.has_hint && meta.storage_offset.hint == 2, label, "symbolic offset changed");
}

static void test_tensor_meta()
{
    expect_tensor(tensor_meta_json(), "tensor real static");
    expect_symbolic_tensor_meta();

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
    expect_tensor_error(tensor_meta_json("[{\"as_name\":\"s0\"},{\"as_int\":4}]"), "$.tensor_meta.sizes[0]", "unknown SymInt tag as_name", "tensor named size");
    expect_tensor_error(tensor_meta_json("[{}, {\"as_int\":4}]"), "$.tensor_meta.sizes[0]", "SymInt union must contain exactly one tag", "tensor empty symint");
    expect_tensor_error(tensor_meta_json("[{\"as_int\":\"3\"},{\"as_int\":4}]"), "$.tensor_meta.sizes[0].as_int", "expected integer", "tensor size type");
    expect_tensor_error(tensor_meta_json("[{\"as_int\":-1},{\"as_int\":4}]"), "$.tensor_meta.sizes[0].as_int", "tensor size must be non-negative", "tensor negative size");
    expect_tensor_error(tensor_meta_json("[{\"as_int\":3},{\"as_int\":4}]", "[{\"as_int\":4}]"), "$.tensor_meta.strides", "rank does not match sizes", "tensor rank mismatch");
    expect_tensor_error(tensor_meta_json("[{\"as_expr\":{\"expr_str\":\"\",\"hint\":null}},{\"as_int\":4}]"), "$.tensor_meta.sizes[0].as_expr.expr_str", "symbolic expression must not be empty", "tensor empty symbolic expression");
    expect_tensor_error(tensor_meta_json("[{\"as_expr\":{\"expr_str\":\"s0\",\"hint\":{\"as_float\":3.0}}},{\"as_int\":4}]"), "$.tensor_meta.sizes[0].as_expr.hint", "SymInt hint must use as_int", "tensor symbolic size hint type");
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

static void test_tiny_program()
{
    ProgramFixture fixture;
    fixture.graph_inputs = "[" + tensor_argument_json("p_linear_weight") + "," + tensor_argument_json("p_linear_bias") + "," + tensor_argument_json("x") + "]";
    fixture.graph_outputs = "[" + tensor_argument_json("relu") + "]";
    fixture.nodes = "[{\"target\":\"torch.ops.aten.linear.default\",\"inputs\":[{\"name\":\"input\",\"arg\":" + tensor_argument_json("x") + ",\"kind\":1},{\"name\":\"weight\",\"arg\":" + tensor_argument_json("p_linear_weight") + ",\"kind\":1},{\"name\":\"bias\",\"arg\":" + tensor_argument_json("p_linear_bias") + ",\"kind\":1}],\"outputs\":[" + tensor_argument_json("linear") + "],\"metadata\":{},\"is_hop_single_tensor_return\":null,\"name\":\"linear\"},{\"target\":\"torch.ops.aten.relu.default\",\"inputs\":[{\"name\":\"self\",\"arg\":" + tensor_argument_json("linear") + ",\"kind\":1}],\"outputs\":[" + tensor_argument_json("relu") + "],\"metadata\":{},\"is_hop_single_tensor_return\":null,\"name\":\"relu\"}]";
    fixture.tensor_values = "{\"p_linear_weight\":" + tensor_meta_json()
                            + ",\"p_linear_bias\":" + tensor_meta_json("[{\"as_int\":3}]", "[{\"as_int\":1}]")
                            + ",\"x\":" + tensor_meta_json("[{\"as_int\":2},{\"as_int\":4}]", "[{\"as_int\":4},{\"as_int\":1}]")
                            + ",\"linear\":" + tensor_meta_json("[{\"as_int\":2},{\"as_int\":3}]", "[{\"as_int\":3},{\"as_int\":1}]")
                            + ",\"relu\":" + tensor_meta_json("[{\"as_int\":2},{\"as_int\":3}]", "[{\"as_int\":3},{\"as_int\":1}]") + "}";
    fixture.input_specs = "[{\"parameter\":{\"arg\":{\"name\":\"p_linear_weight\"},\"parameter_name\":\"linear.weight\"}},{\"parameter\":{\"arg\":{\"name\":\"p_linear_bias\"},\"parameter_name\":\"linear.bias\"}},{\"user_input\":{\"arg\":" + tensor_argument_json("x") + "}}]";
    fixture.output_specs = "[{\"user_output\":{\"arg\":" + tensor_argument_json("relu") + "}}]";

    pnnx::ExportedProgram program;
    if (!parse_program_fixture(fixture, program, "program tiny graph"))
        return;

    check(program.header.schema_major == 8 && program.header.schema_minor == 20, "program tiny graph", "header changed");
    check(program.graph.inputs.size() == 3 && program.graph.inputs[0].name == "p_linear_weight" && program.graph.inputs[2].name == "x", "program tiny graph", "graph inputs changed");
    check(program.graph.nodes.size() == 2, "program tiny graph", "node count changed");
    if (program.graph.nodes.size() == 2)
    {
        check(program.graph.nodes[0].target == "torch.ops.aten.linear.default" && program.graph.nodes[0].has_name && program.graph.nodes[0].name == "linear", "program tiny graph", "linear identity changed");
        check(program.graph.nodes[0].inputs.size() == 3 && program.graph.nodes[0].inputs[1].name == "weight" && program.graph.nodes[0].inputs[1].kind == pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL, "program tiny graph", "linear inputs changed");
        check(program.graph.nodes[1].target == "torch.ops.aten.relu.default" && program.graph.nodes[1].outputs.size() == 1 && program.graph.nodes[1].outputs[0].name == "relu", "program tiny graph", "relu node changed");
    }
    check(program.graph.outputs.size() == 1 && program.graph.outputs[0].name == "relu", "program tiny graph", "graph output changed");
    check(program.graph.tensor_values.size() == 5 && program.graph.tensor_values.find("linear")->second.sizes[1] == 3, "program tiny graph", "tensor values changed");
    check(program.input_specs.size() == 3 && program.input_specs[0].kind == pnnx::EXPORTED_PARAMETER && program.input_specs[0].target == "linear.weight" && program.input_specs[2].kind == pnnx::EXPORTED_USER_INPUT, "program tiny graph", "input signature changed");
    check(program.output_specs.size() == 1 && program.output_specs[0].kind == pnnx::EXPORTED_USER_OUTPUT && program.output_specs[0].arg.name == "relu", "program tiny graph", "output signature changed");
}

static void test_graph_arguments()
{
    pnnx::ExportedArgument argument;
    pnnx::ExportedArgumentKind kind = pnnx::EXPORTED_ARGUMENT_KIND_MISSING;

    if (parse_argument_case("{\"as_none\":true}", ",\"kind\":0", argument, kind, "argument none"))
        check(argument.type == pnnx::EXPORTED_ARGUMENT_NONE && kind == pnnx::EXPORTED_ARGUMENT_KIND_UNKNOWN, "argument none", "none changed");

    if (parse_argument_case(tensor_argument_json("x"), ",\"kind\":1", argument, kind, "argument tensor"))
        check(argument.type == pnnx::EXPORTED_ARGUMENT_TENSOR && argument.name == "x" && kind == pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL, "argument tensor", "tensor changed");

    if (parse_argument_case("{\"as_tensors\":[{\"name\":\"x\"},{\"name\":\"y\"}]}", ",\"kind\":2", argument, kind, "argument tensor list"))
        check(argument.type == pnnx::EXPORTED_ARGUMENT_TENSOR_LIST && argument.tensor_names.size() == 2 && argument.tensor_names[1] == "y" && kind == pnnx::EXPORTED_ARGUMENT_KIND_KEYWORD, "argument tensor list", "tensor list changed");

    if (parse_argument_case("{\"as_int\":-3}", ",\"kind\":null", argument, kind, "argument int"))
        check(argument.type == pnnx::EXPORTED_ARGUMENT_INT && argument.int_value == -3 && kind == pnnx::EXPORTED_ARGUMENT_KIND_MISSING, "argument int", "int changed");

    if (parse_argument_case("{\"as_ints\":[1,-1,3]}", "", argument, kind, "argument int list"))
        check(argument.type == pnnx::EXPORTED_ARGUMENT_INT_LIST && argument.int_values.size() == 3 && argument.int_values[1] == -1 && kind == pnnx::EXPORTED_ARGUMENT_KIND_MISSING, "argument int list", "int list changed");

    if (parse_argument_case("{\"as_float\":1.25}", ",\"kind\":1", argument, kind, "argument float"))
        check(argument.type == pnnx::EXPORTED_ARGUMENT_FLOAT && argument.float_value == 1.25, "argument float", "float changed");

    if (parse_argument_case("{\"as_float\":\"Infinity\"}", ",\"kind\":1", argument, kind, "argument positive infinity"))
        check(argument.type == pnnx::EXPORTED_ARGUMENT_FLOAT && std::isinf(argument.float_value) && argument.float_value > 0.0, "argument positive infinity", "positive infinity changed");

    if (parse_argument_case("{\"as_float\":\"-Infinity\"}", ",\"kind\":1", argument, kind, "argument negative infinity"))
        check(argument.type == pnnx::EXPORTED_ARGUMENT_FLOAT && std::isinf(argument.float_value) && argument.float_value < 0.0, "argument negative infinity", "negative infinity changed");

    if (parse_argument_case("{\"as_float\":\"NaN\"}", ",\"kind\":1", argument, kind, "argument nan"))
        check(argument.type == pnnx::EXPORTED_ARGUMENT_FLOAT && std::isnan(argument.float_value), "argument nan", "nan changed");

    if (parse_argument_case("{\"as_floats\":[1.5,-2.25]}", ",\"kind\":1", argument, kind, "argument float list"))
        check(argument.type == pnnx::EXPORTED_ARGUMENT_FLOAT_LIST && argument.float_values.size() == 2 && argument.float_values[1] == -2.25, "argument float list", "float list changed");

    if (parse_argument_case("{\"as_floats\":[\"Infinity\",\"-Infinity\",\"NaN\"]}", ",\"kind\":1", argument, kind, "argument special float list"))
        check(argument.type == pnnx::EXPORTED_ARGUMENT_FLOAT_LIST && argument.float_values.size() == 3 && std::isinf(argument.float_values[0]) && argument.float_values[0] > 0.0 && std::isinf(argument.float_values[1]) && argument.float_values[1] < 0.0 && std::isnan(argument.float_values[2]), "argument special float list", "special float list changed");

    if (parse_argument_case("{\"as_bool\":true}", ",\"kind\":1", argument, kind, "argument bool"))
        check(argument.type == pnnx::EXPORTED_ARGUMENT_BOOL && argument.bool_value, "argument bool", "bool changed");

    if (parse_argument_case("{\"as_bools\":[true,false]}", ",\"kind\":1", argument, kind, "argument bool list"))
        check(argument.type == pnnx::EXPORTED_ARGUMENT_BOOL_LIST && argument.bool_values.size() == 2 && !argument.bool_values[1], "argument bool list", "bool list changed");

    if (parse_argument_case("{\"as_string\":\"nearest\"}", ",\"kind\":1", argument, kind, "argument string"))
        check(argument.type == pnnx::EXPORTED_ARGUMENT_STRING && argument.string_value == "nearest", "argument string", "string changed");

    if (parse_argument_case("{\"as_strings\":[\"a\",\"b\"]}", ",\"kind\":1", argument, kind, "argument string list"))
        check(argument.type == pnnx::EXPORTED_ARGUMENT_STRING_LIST && argument.string_values.size() == 2 && argument.string_values[1] == "b", "argument string list", "string list changed");

    if (parse_argument_case("{\"as_scalar_type\":7}", ",\"kind\":1", argument, kind, "argument scalar type"))
        check(argument.type == pnnx::EXPORTED_ARGUMENT_SCALAR_TYPE && argument.enum_value == 7, "argument scalar type", "scalar type changed");

    if (parse_argument_case("{\"as_memory_format\":2}", ",\"kind\":1", argument, kind, "argument memory format"))
        check(argument.type == pnnx::EXPORTED_ARGUMENT_MEMORY_FORMAT && argument.enum_value == 2, "argument memory format", "memory format changed");

    if (parse_argument_case("{\"as_layout\":7}", ",\"kind\":1", argument, kind, "argument layout"))
        check(argument.type == pnnx::EXPORTED_ARGUMENT_LAYOUT && argument.enum_value == 7, "argument layout", "layout changed");

    if (parse_argument_case("{\"as_device\":{\"type\":\"cuda\",\"index\":2}}", ",\"kind\":1", argument, kind, "argument device"))
        check(argument.type == pnnx::EXPORTED_ARGUMENT_DEVICE && argument.device_value.type == "cuda" && argument.device_value.has_index && argument.device_value.index == 2, "argument device", "device changed");

    if (parse_argument_case("{\"as_sym_int\":{\"as_int\":5}}", ",\"kind\":1", argument, kind, "argument static sym int"))
        check(argument.type == pnnx::EXPORTED_ARGUMENT_INT && argument.int_value == 5, "argument static sym int", "static sym int changed");

    if (parse_argument_case("{\"as_sym_ints\":[{\"as_int\":2},{\"as_int\":-1}]}", ",\"kind\":1", argument, kind, "argument static sym ints"))
        check(argument.type == pnnx::EXPORTED_ARGUMENT_INT_LIST && argument.int_values.size() == 2 && argument.int_values[1] == -1, "argument static sym ints", "static sym ints changed");

    if (parse_argument_case("{\"as_sym_ints\":[]}", ",\"kind\":1", argument, kind, "argument empty sym ints"))
        check(argument.type == pnnx::EXPORTED_ARGUMENT_INT_LIST && argument.int_values.empty(), "argument empty sym ints", "empty sym ints changed");

    {
        ProgramFixture fixture;
        fixture.sym_int_values = "{\"s0\":{\"as_expr\":{\"expr_str\":\"Symbol('u0', integer=True)\",\"hint\":null}},\"s1\":{\"as_expr\":{\"expr_str\":\"Symbol('u1', integer=True)\",\"hint\":null}}}";
        fixture.nodes = "[{\"target\":\"torch.ops.aten.reshape.default\",\"inputs\":[{\"name\":\"self\",\"arg\":" + tensor_argument_json("x") + ",\"kind\":1},{\"name\":\"shape\",\"arg\":{\"as_sym_ints\":[{\"as_int\":3},{\"as_name\":\"s0\"},{\"as_int\":16},{\"as_name\":\"s1\"}]},\"kind\":1}],\"outputs\":[" + tensor_argument_json("y") + "],\"metadata\":{},\"is_hop_single_tensor_return\":null,\"name\":\"y\"}]";

        pnnx::ExportedProgram program;
        if (parse_program_fixture(fixture, program, "argument dynamic sym ints"))
        {
            argument = program.graph.nodes[0].inputs[1].arg;
            check(argument.type == pnnx::EXPORTED_ARGUMENT_SYMBOLIC_INT_LIST, "argument dynamic sym ints", "dynamic SymInt list type changed");
            check(argument.symbolic_int_values.size() == 4, "argument dynamic sym ints", "dynamic SymInt list length changed");
            if (argument.symbolic_int_values.size() == 4)
            {
                check(argument.symbolic_int_values[0].type == pnnx::EXPORTED_SYM_INT_LIST_STATIC && argument.symbolic_int_values[0].value == 3, "argument dynamic sym ints", "first static value changed");
                check(argument.symbolic_int_values[1].type == pnnx::EXPORTED_SYM_INT_LIST_SYMBOLIC && argument.symbolic_int_values[1].name == "s0", "argument dynamic sym ints", "first symbolic value changed");
                check(argument.symbolic_int_values[2].type == pnnx::EXPORTED_SYM_INT_LIST_STATIC && argument.symbolic_int_values[2].value == 16, "argument dynamic sym ints", "second static value changed");
                check(argument.symbolic_int_values[3].type == pnnx::EXPORTED_SYM_INT_LIST_SYMBOLIC && argument.symbolic_int_values[3].name == "s1", "argument dynamic sym ints", "second symbolic value changed");
            }
        }

        fixture.sym_int_values = "{\"s0\":{\"as_expr\":{\"expr_str\":\"Symbol('u0', integer=True)\",\"hint\":null}}}";
        expect_program_error(fixture, "$.graph_module.graph.sym_int_values.s1", "missing symbolic int value", "argument dynamic sym ints missing symbol");
    }

    if (parse_argument_case("{\"as_sym_float\":{\"as_float\":1.5}}", ",\"kind\":1", argument, kind, "argument static sym float"))
        check(argument.type == pnnx::EXPORTED_ARGUMENT_FLOAT && argument.float_value == 1.5, "argument static sym float", "static sym float changed");

    if (parse_argument_case("{\"as_sym_float\":{\"as_float\":\"-Infinity\"}}", ",\"kind\":1", argument, kind, "argument static sym negative infinity"))
        check(argument.type == pnnx::EXPORTED_ARGUMENT_FLOAT && std::isinf(argument.float_value) && argument.float_value < 0.0, "argument static sym negative infinity", "static sym negative infinity changed");

    if (parse_argument_case("{\"as_sym_floats\":[{\"as_float\":1.5},{\"as_float\":2.5}]}", ",\"kind\":1", argument, kind, "argument static sym floats"))
        check(argument.type == pnnx::EXPORTED_ARGUMENT_FLOAT_LIST && argument.float_values.size() == 2 && argument.float_values[1] == 2.5, "argument static sym floats", "static sym floats changed");

    if (parse_argument_case("{\"as_sym_floats\":[{\"as_float\":\"Infinity\"},{\"as_float\":\"NaN\"}]}", ",\"kind\":1", argument, kind, "argument static sym special floats"))
        check(argument.type == pnnx::EXPORTED_ARGUMENT_FLOAT_LIST && argument.float_values.size() == 2 && std::isinf(argument.float_values[0]) && argument.float_values[0] > 0.0 && std::isnan(argument.float_values[1]), "argument static sym special floats", "static sym special floats changed");

    if (parse_argument_case("{\"as_sym_bool\":{\"as_bool\":false}}", ",\"kind\":1", argument, kind, "argument static sym bool"))
        check(argument.type == pnnx::EXPORTED_ARGUMENT_BOOL && !argument.bool_value, "argument static sym bool", "static sym bool changed");

    if (parse_argument_case("{\"as_sym_bools\":[{\"as_bool\":true},{\"as_bool\":false}]}", ",\"kind\":1", argument, kind, "argument static sym bools"))
        check(argument.type == pnnx::EXPORTED_ARGUMENT_BOOL_LIST && argument.bool_values.size() == 2 && !argument.bool_values[1], "argument static sym bools", "static sym bools changed");

    {
        ProgramFixture fixture;
        fixture.sym_int_values = "{\"s0\":{\"as_expr\":{\"expr_str\":\"Symbol('u0', integer=True)\",\"hint\":null}}}";
        fixture.nodes = "[{\"target\":\"torch.ops.aten.clone.default\",\"inputs\":[{\"name\":\"value\",\"arg\":{\"as_sym_int\":{\"as_name\":\"s0\"}},\"kind\":1}],\"outputs\":[" + tensor_argument_json("y") + "],\"metadata\":{},\"is_hop_single_tensor_return\":null,\"name\":\"y\"}]";
        pnnx::ExportedProgram program;
        if (parse_program_fixture(fixture, program, "argument dynamic sym int"))
        {
            argument = program.graph.nodes[0].inputs[0].arg;
            check(argument.type == pnnx::EXPORTED_ARGUMENT_SYMBOLIC_INT && argument.name == "s0", "argument dynamic sym int", "dynamic sym int classification changed");
        }
    }

    if (parse_argument_case("{\"as_custom_obj\":{\"name\":\"obj\",\"class_fqn\":\"pkg.Type\"}}", ",\"kind\":1", argument, kind, "argument custom object"))
        check(argument.type == pnnx::EXPORTED_ARGUMENT_UNSUPPORTED && argument.unsupported_tag == "as_custom_obj" && argument.name == "obj" && argument.string_value == "pkg.Type", "argument custom object", "custom object classification changed");

    if (parse_argument_case("{\"as_operator\":\"torch.ops.aten.relu.default\"}", ",\"kind\":1", argument, kind, "argument operator"))
        check(argument.type == pnnx::EXPORTED_ARGUMENT_UNSUPPORTED && argument.unsupported_tag == "as_operator" && argument.string_value == "torch.ops.aten.relu.default", "argument operator", "operator classification changed");

    if (parse_argument_case(graph_argument_json("submod_1"), ",\"kind\":1", argument, kind, "argument graph"))
    {
        check(argument.type == pnnx::EXPORTED_ARGUMENT_GRAPH, "argument graph", "graph classification changed");
        check(argument.graph_name == "submod_1", "argument graph", "graph name changed");
        check(argument.graph_value && argument.graph_value->inputs.size() == 1 && argument.graph_value->nodes.size() == 1 && argument.graph_value->outputs.size() == 1, "argument graph", "nested graph shape changed");
        check(argument.graph_value && argument.graph_value->nodes[0].target == "torch.ops.aten.relu.default", "argument graph", "nested graph target changed");
    }

    if (parse_argument_case("{\"as_optional_tensor\":{\"as_none\":true}}", ",\"kind\":1", argument, kind, "argument optional tensor"))
        check(argument.type == pnnx::EXPORTED_ARGUMENT_UNSUPPORTED && argument.unsupported_tag == "as_optional_tensor", "argument optional tensor", "optional tensor classification changed");

    if (parse_argument_case("{\"as_optional_tensors\":[]}", ",\"kind\":1", argument, kind, "argument optional tensor list"))
        check(argument.type == pnnx::EXPORTED_ARGUMENT_UNSUPPORTED && argument.unsupported_tag == "as_optional_tensors", "argument optional tensor list", "optional tensor list classification changed");

    if (parse_argument_case("{\"as_complex\":{\"real\":1.0,\"imag\":-2.0}}", ",\"kind\":1", argument, kind, "argument complex"))
        check(argument.type == pnnx::EXPORTED_ARGUMENT_COMPLEX && argument.complex_real_value == 1.0 && argument.complex_imag_value == -2.0, "argument complex", "complex value changed");

    if (parse_argument_case("{\"as_nested_tensors\":[]}", ",\"kind\":1", argument, kind, "argument nested tensors"))
        check(argument.type == pnnx::EXPORTED_ARGUMENT_UNSUPPORTED && argument.unsupported_tag == "as_nested_tensors", "argument nested tensors", "nested tensors classification changed");

    if (parse_argument_case("{\"as_int_lists\":[[1,2]]}", ",\"kind\":1", argument, kind, "argument int lists"))
        check(argument.type == pnnx::EXPORTED_ARGUMENT_UNSUPPORTED && argument.unsupported_tag == "as_int_lists", "argument int lists", "int lists classification changed");

    if (parse_argument_case("{\"as_float_lists\":[[1.0,2.0]]}", ",\"kind\":1", argument, kind, "argument float lists"))
        check(argument.type == pnnx::EXPORTED_ARGUMENT_UNSUPPORTED && argument.unsupported_tag == "as_float_lists", "argument float lists", "float lists classification changed");

    if (parse_argument_case("{\"as_string_to_argument\":{}}", ",\"kind\":1", argument, kind, "argument string map"))
        check(argument.type == pnnx::EXPORTED_ARGUMENT_UNSUPPORTED && argument.unsupported_tag == "as_string_to_argument", "argument string map", "string map classification changed");

    ProgramFixture fixture;
    fixture.nodes = "[{\"target\":\"torch.ops.aten.clone.default\",\"inputs\":[{\"name\":\"value\",\"arg\":{\"as_future\":1},\"kind\":1}],\"outputs\":[" + tensor_argument_json("y") + "],\"metadata\":{},\"is_hop_single_tensor_return\":null,\"name\":\"y\"}]";
    expect_program_error(fixture, "$.graph_module.graph.nodes[0].inputs[0].arg", "unknown argument tag as_future", "argument unknown tag");

    fixture = ProgramFixture();
    fixture.nodes = "[{\"target\":\"torch.ops.aten.clone.default\",\"inputs\":[{\"name\":\"value\",\"arg\":{\"as_int\":1,\"as_bool\":true},\"kind\":1}],\"outputs\":[" + tensor_argument_json("y") + "],\"metadata\":{},\"is_hop_single_tensor_return\":null,\"name\":\"y\"}]";
    expect_program_error(fixture, "$.graph_module.graph.nodes[0].inputs[0].arg", "argument union must contain exactly one tag", "argument multiple tags");

    fixture = ProgramFixture();
    fixture.nodes = "[{\"target\":\"torch.ops.aten.clone.default\",\"inputs\":[{\"name\":\"value\",\"arg\":{\"as_ints\":[1,true]},\"kind\":1}],\"outputs\":[" + tensor_argument_json("y") + "],\"metadata\":{},\"is_hop_single_tensor_return\":null,\"name\":\"y\"}]";
    expect_program_error(fixture, "$.graph_module.graph.nodes[0].inputs[0].arg.as_ints[1]", "expected integer", "argument list item type");

    fixture = ProgramFixture();
    fixture.nodes = "[{\"target\":\"torch.ops.aten.reshape.default\",\"inputs\":[{\"name\":\"self\",\"arg\":" + tensor_argument_json("x") + ",\"kind\":1},{\"name\":\"shape\",\"arg\":{\"as_sym_ints\":[{\"as_float\":1.0}]},\"kind\":1}],\"outputs\":[" + tensor_argument_json("y") + "],\"metadata\":{},\"is_hop_single_tensor_return\":null,\"name\":\"y\"}]";
    expect_program_error(fixture, "$.graph_module.graph.nodes[0].inputs[1].arg.as_sym_ints[0]", "unknown symbolic argument tag as_float", "argument dynamic sym ints element tag");

    fixture = ProgramFixture();
    fixture.nodes = "[{\"target\":\"torch.ops.aten.reshape.default\",\"inputs\":[{\"name\":\"self\",\"arg\":" + tensor_argument_json("x") + ",\"kind\":1},{\"name\":\"shape\",\"arg\":{\"as_sym_ints\":[{\"as_name\":\"\"}]},\"kind\":1}],\"outputs\":[" + tensor_argument_json("y") + "],\"metadata\":{},\"is_hop_single_tensor_return\":null,\"name\":\"y\"}]";
    expect_program_error(fixture, "$.graph_module.graph.nodes[0].inputs[1].arg.as_sym_ints[0].as_name", "symbolic argument name must not be empty", "argument dynamic sym ints empty symbol");

    fixture = ProgramFixture();
    fixture.nodes = "[{\"target\":\"torch.ops.aten.reshape.default\",\"inputs\":[{\"name\":\"self\",\"arg\":" + tensor_argument_json("x") + ",\"kind\":1},{\"name\":\"shape\",\"arg\":{\"as_sym_ints\":[{\"as_int\":\"3\"}]},\"kind\":1}],\"outputs\":[" + tensor_argument_json("y") + "],\"metadata\":{},\"is_hop_single_tensor_return\":null,\"name\":\"y\"}]";
    expect_program_error(fixture, "$.graph_module.graph.nodes[0].inputs[1].arg.as_sym_ints[0].as_int", "expected integer", "argument dynamic sym ints static type");

    fixture = ProgramFixture();
    fixture.nodes = "[{\"target\":\"torch.ops.aten.clone.default\",\"inputs\":[{\"name\":\"value\",\"arg\":{\"as_float\":\"infinity\"},\"kind\":1}],\"outputs\":[" + tensor_argument_json("y") + "],\"metadata\":{},\"is_hop_single_tensor_return\":null,\"name\":\"y\"}]";
    expect_program_error(fixture, "$.graph_module.graph.nodes[0].inputs[0].arg.as_float", "unknown special float value", "argument invalid special float");

    fixture = ProgramFixture();
    fixture.nodes = "[{\"target\":\"torch.ops.aten.clone.default\",\"inputs\":[{\"name\":\"value\",\"arg\":{\"as_complex\":{\"imag\":2.0}},\"kind\":1}],\"outputs\":[" + tensor_argument_json("y") + "],\"metadata\":{},\"is_hop_single_tensor_return\":null,\"name\":\"y\"}]";
    expect_program_error(fixture, "$.graph_module.graph.nodes[0].inputs[0].arg.as_complex.real", "missing required field", "argument complex missing real");

    fixture = ProgramFixture();
    fixture.nodes = "[{\"target\":\"torch.ops.aten.clone.default\",\"inputs\":[{\"name\":\"value\",\"arg\":{\"as_complex\":{\"real\":1.0}},\"kind\":1}],\"outputs\":[" + tensor_argument_json("y") + "],\"metadata\":{},\"is_hop_single_tensor_return\":null,\"name\":\"y\"}]";
    expect_program_error(fixture, "$.graph_module.graph.nodes[0].inputs[0].arg.as_complex.imag", "missing required field", "argument complex missing imag");

    fixture = ProgramFixture();
    fixture.nodes = "[{\"target\":\"torch.ops.aten.clone.default\",\"inputs\":[{\"name\":\"value\",\"arg\":{\"as_complex\":{\"real\":1,\"imag\":2.0}},\"kind\":1}],\"outputs\":[" + tensor_argument_json("y") + "],\"metadata\":{},\"is_hop_single_tensor_return\":null,\"name\":\"y\"}]";
    expect_program_error(fixture, "$.graph_module.graph.nodes[0].inputs[0].arg.as_complex.real", "expected float", "argument complex real type");

    fixture = ProgramFixture();
    fixture.nodes = "[{\"target\":\"torch.ops.aten.clone.default\",\"inputs\":[{\"name\":\"value\",\"arg\":{\"as_complex\":{\"real\":1.0,\"imag\":false}},\"kind\":1}],\"outputs\":[" + tensor_argument_json("y") + "],\"metadata\":{},\"is_hop_single_tensor_return\":null,\"name\":\"y\"}]";
    expect_program_error(fixture, "$.graph_module.graph.nodes[0].inputs[0].arg.as_complex.imag", "expected float", "argument complex imag type");

    fixture = ProgramFixture();
    fixture.nodes = "[{\"target\":\"torch.ops.aten.clone.default\",\"inputs\":[{\"name\":\"value\",\"arg\":{\"as_int\":1},\"kind\":3}],\"outputs\":[" + tensor_argument_json("y") + "],\"metadata\":{},\"is_hop_single_tensor_return\":null,\"name\":\"y\"}]";
    expect_program_error(fixture, "$.graph_module.graph.nodes[0].inputs[0].kind", "unknown argument kind", "argument invalid kind");

    fixture = ProgramFixture();
    fixture.nodes = "[{\"target\":\"torch.ops.higher_order.wrap_with_set_grad_enabled\",\"inputs\":[{\"name\":\"\",\"arg\":{\"as_bool\":false},\"kind\":1},{\"name\":\"\",\"arg\":" + graph_argument_json("submod_1") + ",\"kind\":1},{\"name\":\"\",\"arg\":" + tensor_argument_json("x") + ",\"kind\":1}],\"outputs\":[" + tensor_argument_json("y") + "],\"metadata\":{},\"is_hop_single_tensor_return\":null,\"name\":\"wrapped\"}]";
    pnnx::ExportedProgram positional_program;
    if (parse_program_fixture(fixture, positional_program, "empty positional argument name"))
    {
        check(positional_program.graph.nodes[0].inputs.size() == 3, "empty positional argument name", "input count changed");
        check(positional_program.graph.nodes[0].inputs[0].name.empty() && positional_program.graph.nodes[0].inputs[1].name.empty(), "empty positional argument name", "empty names changed");
    }

    fixture = ProgramFixture();
    fixture.nodes = "[{\"target\":\"torch.ops.aten.clone.default\",\"inputs\":[{\"name\":\"\",\"arg\":{\"as_int\":1},\"kind\":2}],\"outputs\":[" + tensor_argument_json("y") + "],\"metadata\":{},\"is_hop_single_tensor_return\":null,\"name\":\"y\"}]";
    expect_program_error(fixture, "$.graph_module.graph.nodes[0].inputs[0].name", "name must not be empty", "empty keyword argument name");

    fixture = ProgramFixture();
    fixture.nodes = "[{\"target\":\"torch.ops.higher_order.wrap_with_set_grad_enabled\",\"inputs\":[{\"name\":\"\",\"arg\":{\"as_graph\":{\"graph\":{}}},\"kind\":1}],\"outputs\":[" + tensor_argument_json("y") + "],\"metadata\":{},\"is_hop_single_tensor_return\":null,\"name\":\"wrapped\"}]";
    expect_program_error(fixture, "$.graph_module.graph.nodes[0].inputs[0].arg.as_graph.name", "missing required field", "graph argument missing name");

    fixture = ProgramFixture();
    fixture.nodes = "[{\"target\":\"torch.ops.higher_order.wrap_with_set_grad_enabled\",\"inputs\":[{\"name\":\"\",\"arg\":{\"as_graph\":{\"name\":\"\",\"graph\":{}}},\"kind\":1}],\"outputs\":[" + tensor_argument_json("y") + "],\"metadata\":{},\"is_hop_single_tensor_return\":null,\"name\":\"wrapped\"}]";
    expect_program_error(fixture, "$.graph_module.graph.nodes[0].inputs[0].arg.as_graph.name", "graph name must not be empty", "graph argument empty name");

    fixture = ProgramFixture();
    fixture.nodes = "[{\"target\":\"torch.ops.higher_order.wrap_with_set_grad_enabled\",\"inputs\":[{\"name\":\"\",\"arg\":{\"as_graph\":{\"name\":\"submod\"}},\"kind\":1}],\"outputs\":[" + tensor_argument_json("y") + "],\"metadata\":{},\"is_hop_single_tensor_return\":null,\"name\":\"wrapped\"}]";
    expect_program_error(fixture, "$.graph_module.graph.nodes[0].inputs[0].arg.as_graph.graph", "missing required field", "graph argument missing graph");
}

static void test_graph_signatures()
{
    ProgramFixture fixture;
    fixture.graph_inputs = "[" + tensor_argument_json("p") + "," + tensor_argument_json("b") + "," + tensor_argument_json("c") + "," + tensor_argument_json("x") + "]";
    fixture.tensor_values = "{\"p\":" + tensor_meta_json("[{\"as_int\":2}]", "[{\"as_int\":1}]")
                            + ",\"b\":" + tensor_meta_json("[{\"as_int\":2}]", "[{\"as_int\":1}]")
                            + ",\"c\":" + tensor_meta_json("[{\"as_int\":2}]", "[{\"as_int\":1}]")
                            + ",\"x\":" + tensor_meta_json("[{\"as_int\":2}]", "[{\"as_int\":1}]")
                            + ",\"y\":" + tensor_meta_json("[{\"as_int\":2}]", "[{\"as_int\":1}]") + "}";
    fixture.input_specs = "[{\"parameter\":{\"arg\":{\"name\":\"p\"},\"parameter_name\":\"weight\"}},{\"buffer\":{\"arg\":{\"name\":\"b\"},\"buffer_name\":\"running\",\"persistent\":false}},{\"tensor_constant\":{\"arg\":{\"name\":\"c\"},\"tensor_constant_name\":\"constant\"}},{\"user_input\":{\"arg\":" + tensor_argument_json("x") + "}}]";

    pnnx::ExportedProgram program;
    if (parse_program_fixture(fixture, program, "signature state inputs"))
    {
        check(program.input_specs.size() == 4, "signature state inputs", "state input count changed");
        if (program.input_specs.size() == 4)
        {
            check(program.input_specs[0].kind == pnnx::EXPORTED_PARAMETER && program.input_specs[0].target == "weight", "signature state inputs", "parameter changed");
            check(program.input_specs[1].kind == pnnx::EXPORTED_BUFFER && program.input_specs[1].target == "running" && !program.input_specs[1].persistent, "signature state inputs", "buffer changed");
            check(program.input_specs[2].kind == pnnx::EXPORTED_TENSOR_CONSTANT && program.input_specs[2].target == "constant", "signature state inputs", "tensor constant changed");
            check(program.input_specs[3].kind == pnnx::EXPORTED_USER_INPUT && program.input_specs[3].arg.name == "x", "signature state inputs", "user input changed");
        }
    }

    fixture = ProgramFixture();
    fixture.graph_inputs = "[" + tensor_argument_json("x") + ",{\"as_int\":7}]";
    fixture.input_specs = "[{\"user_input\":{\"arg\":" + tensor_argument_json("x") + "}},{\"constant_input\":{\"name\":\"n\",\"value\":{\"as_int\":7}}}]";
    if (parse_program_fixture(fixture, program, "signature constant input"))
        check(program.input_specs.size() == 2 && program.input_specs[1].kind == pnnx::EXPORTED_CONSTANT_INPUT && program.input_specs[1].arg.name == "n" && program.input_specs[1].arg.int_value == 7, "signature constant input", "constant input changed");

    fixture = ProgramFixture();
    fixture.graph_inputs = "[{\"as_custom_obj\":{\"name\":\"obj\",\"class_fqn\":\"pkg.Type\"}}]";
    fixture.custom_obj_values = "{\"obj\":{\"name\":\"obj\",\"class_fqn\":\"pkg.Type\"}}";
    fixture.input_specs = "[{\"custom_obj\":{\"arg\":{\"name\":\"obj\",\"class_fqn\":\"pkg.Type\"},\"custom_obj_name\":\"state.obj\"}}]";
    if (parse_program_fixture(fixture, program, "signature custom object"))
        check(program.input_specs.size() == 1 && program.input_specs[0].kind == pnnx::EXPORTED_CUSTOM_OBJ && program.input_specs[0].target == "state.obj" && program.graph.custom_obj_values.size() == 1, "signature custom object", "custom object input changed");

    fixture = ProgramFixture();
    fixture.graph_inputs = "[{\"as_none\":true}]";
    fixture.input_specs = "[{\"token\":{\"arg\":{\"name\":\"token0\"}}}]";
    if (parse_program_fixture(fixture, program, "signature token input"))
        check(program.input_specs.size() == 1 && program.input_specs[0].kind == pnnx::EXPORTED_TOKEN && program.input_specs[0].arg.name == "token0", "signature token input", "token input changed");

    fixture = ProgramFixture();
    fixture.output_specs = "[{\"buffer_mutation\":{\"arg\":{\"name\":\"y\"},\"buffer_name\":\"running\"}}]";
    if (parse_program_fixture(fixture, program, "signature buffer mutation"))
        check(program.output_specs[0].kind == pnnx::EXPORTED_BUFFER_MUTATION && program.output_specs[0].target == "running", "signature buffer mutation", "buffer mutation changed");

    fixture.output_specs = "[{\"parameter_mutation\":{\"arg\":{\"name\":\"y\"},\"parameter_name\":\"weight\"}}]";
    if (parse_program_fixture(fixture, program, "signature parameter mutation"))
        check(program.output_specs[0].kind == pnnx::EXPORTED_PARAMETER_MUTATION && program.output_specs[0].target == "weight", "signature parameter mutation", "parameter mutation changed");

    fixture.output_specs = "[{\"gradient_to_parameter\":{\"arg\":{\"name\":\"y\"},\"parameter_name\":\"weight\"}}]";
    if (parse_program_fixture(fixture, program, "signature parameter gradient"))
        check(program.output_specs[0].kind == pnnx::EXPORTED_GRADIENT_TO_PARAMETER && program.output_specs[0].target == "weight", "signature parameter gradient", "parameter gradient changed");

    fixture.output_specs = "[{\"gradient_to_user_input\":{\"arg\":{\"name\":\"y\"},\"user_input_name\":\"x\"}}]";
    if (parse_program_fixture(fixture, program, "signature user input gradient"))
        check(program.output_specs[0].kind == pnnx::EXPORTED_GRADIENT_TO_USER_INPUT && program.output_specs[0].target == "x", "signature user input gradient", "user input gradient changed");

    fixture.output_specs = "[{\"user_input_mutation\":{\"arg\":{\"name\":\"y\"},\"user_input_name\":\"x\"}}]";
    if (parse_program_fixture(fixture, program, "signature user input mutation"))
        check(program.output_specs[0].kind == pnnx::EXPORTED_USER_INPUT_MUTATION && program.output_specs[0].target == "x", "signature user input mutation", "user input mutation changed");

    fixture.output_specs = "[{\"loss_output\":{\"arg\":{\"name\":\"y\"}}}]";
    if (parse_program_fixture(fixture, program, "signature loss output"))
        check(program.output_specs[0].kind == pnnx::EXPORTED_LOSS_OUTPUT && program.output_specs[0].arg.name == "y", "signature loss output", "loss output changed");

    fixture.output_specs = "[{\"token\":{\"arg\":{\"name\":\"token1\"}}}]";
    if (parse_program_fixture(fixture, program, "signature token output"))
        check(program.output_specs[0].kind == pnnx::EXPORTED_OUTPUT_TOKEN && program.output_specs[0].arg.name == "token1", "signature token output", "token output changed");
}

static void test_output_tree_specs()
{
    ProgramFixture fixture;
    pnnx::ExportedProgram program;
    if (parse_program_fixture(fixture, program, "output treespec leaf"))
    {
        check(program.output_tree_spec.type == pnnx::EXPORTED_TREE_SPEC_LEAF, "output treespec leaf", "leaf type changed");
        check(program.output_tree_spec.children.empty(), "output treespec leaf", "leaf has children");
    }

    fixture = ProgramFixture();
    fixture.module_call_graph = "[]";
    if (parse_program_fixture(fixture, program, "output treespec legacy flat"))
        check(program.output_tree_spec.type == pnnx::EXPORTED_TREE_SPEC_NONE, "output treespec legacy flat", "missing root did not preserve flat outputs");

    fixture = ProgramFixture();
    fixture.module_call_graph = module_call_graph_json("[1,{\"type\":\"builtins.tuple\",\"context\":\"null\",\"children_spec\":[{\"type\":null,\"context\":null,\"children_spec\":[]}]}]");
    if (parse_program_fixture(fixture, program, "output treespec one tuple"))
    {
        check(program.output_tree_spec.type == pnnx::EXPORTED_TREE_SPEC_TUPLE, "output treespec one tuple", "tuple type changed");
        check(program.output_tree_spec.children.size() == 1 && program.output_tree_spec.children[0].type == pnnx::EXPORTED_TREE_SPEC_LEAF, "output treespec one tuple", "tuple child changed");
    }

    fixture = ProgramFixture();
    fixture.graph_outputs = "[" + tensor_argument_json("y") + "," + tensor_argument_json("y") + "," + tensor_argument_json("y") + "]";
    fixture.output_specs = "[{\"user_output\":{\"arg\":" + tensor_argument_json("y") + "}},{\"user_output\":{\"arg\":" + tensor_argument_json("y") + "}},{\"user_output\":{\"arg\":" + tensor_argument_json("y") + "}}]";
    fixture.module_call_graph = module_call_graph_json("[1,{\"type\":\"builtins.tuple\",\"context\":\"null\",\"children_spec\":[{\"type\":null,\"context\":null,\"children_spec\":[]},{\"type\":\"builtins.list\",\"context\":\"null\",\"children_spec\":[{\"type\":null,\"context\":null,\"children_spec\":[]},{\"type\":null,\"context\":null,\"children_spec\":[]}]}]}]");
    if (parse_program_fixture(fixture, program, "output treespec nested"))
    {
        check(program.output_tree_spec.type == pnnx::EXPORTED_TREE_SPEC_TUPLE && program.output_tree_spec.children.size() == 2, "output treespec nested", "outer tuple changed");
        check(program.output_tree_spec.children.size() == 2 && program.output_tree_spec.children[1].type == pnnx::EXPORTED_TREE_SPEC_LIST && program.output_tree_spec.children[1].children.size() == 2, "output treespec nested", "nested list changed");
    }

    fixture = ProgramFixture();
    fixture.module_call_graph = "[{\"fqn\":\"\",\"signature\":{\"out_spec\":7}}]";
    expect_program_error(fixture, "$.graph_module.module_call_graph[0].signature.out_spec", "expected string", "output treespec string type");

    fixture = ProgramFixture();
    fixture.module_call_graph = module_call_graph_json("not-json");
    expect_program_error(fixture, "$.graph_module.module_call_graph[0].signature.out_spec", "invalid output treespec JSON", "output treespec invalid json");

    fixture = ProgramFixture();
    fixture.module_call_graph = module_call_graph_json("[2,{\"type\":null,\"context\":null,\"children_spec\":[]}]");
    expect_program_error(fixture, "$.graph_module.module_call_graph[0].signature.out_spec", "unsupported output treespec protocol 2", "output treespec protocol");

    fixture = ProgramFixture();
    fixture.module_call_graph = module_call_graph_json("[1,{\"type\":\"builtins.dict\",\"context\":\"[]\",\"children_spec\":[]}]");
    expect_program_error(fixture, "$.graph_module.module_call_graph[0].signature.out_spec", "unsupported output treespec type builtins.dict", "output treespec unsupported type");

    fixture = ProgramFixture();
    fixture.module_call_graph = module_call_graph_json("[1,{\"type\":\"builtins.tuple\",\"context\":null,\"children_spec\":[]}]");
    expect_program_error(fixture, "$.graph_module.module_call_graph[0].signature.out_spec", "container context must be the string null", "output treespec tuple context");

    fixture = ProgramFixture();
    fixture.module_call_graph = module_call_graph_json("[1,{\"type\":null,\"context\":null,\"children_spec\":[{\"type\":null,\"context\":null,\"children_spec\":[]}]}]");
    expect_program_error(fixture, "$.graph_module.module_call_graph[0].signature.out_spec", "leaf must not have children", "output treespec leaf children");

    fixture = ProgramFixture();
    fixture.module_call_graph = "[{\"fqn\":\"\",\"signature\":null}]";
    expect_program_error(fixture, "$.graph_module.module_call_graph[0].signature", "root module signature is required", "output treespec root signature");

    fixture = ProgramFixture();
    fixture.module_call_graph = module_call_graph_json("[1,{\"type\":null,\"context\":null,\"children_spec\":[]}]");
    fixture.module_call_graph.insert(fixture.module_call_graph.size() - 1, ",{\"fqn\":\"\",\"signature\":null}");
    expect_program_error(fixture, "$.graph_module.module_call_graph[1].fqn", "duplicate root module entry", "output treespec duplicate root");

    fixture = ProgramFixture();
    fixture.module_call_graph = module_call_graph_json("[1,{\"type\":\"builtins.tuple\",\"context\":\"null\",\"children_spec\":[{\"type\":null,\"context\":null,\"children_spec\":[]},{\"type\":null,\"context\":null,\"children_spec\":[]}]}]");
    expect_program_error(fixture, "$.graph_module.module_call_graph[0].signature.out_spec", "leaf count does not match graph outputs", "output treespec leaf count");
}

static void test_program_errors()
{
    ProgramFixture fixture;
    fixture.graph_inputs = "[" + tensor_argument_json("missing") + "]";
    expect_program_error(fixture, "$.graph_module.graph.tensor_values.missing", "missing tensor metadata", "program missing tensor value");

    fixture = ProgramFixture();
    fixture.nodes = "[{\"target\":\"torch.ops.aten.cat.default\",\"inputs\":[{\"name\":\"tensors\",\"arg\":{\"as_tensors\":[{\"name\":\"x\"},{\"name\":\"missing\"}]},\"kind\":1}],\"outputs\":[" + tensor_argument_json("y") + "],\"metadata\":{},\"is_hop_single_tensor_return\":null,\"name\":\"y\"}]";
    expect_program_error(fixture, "$.graph_module.graph.tensor_values.missing", "missing tensor metadata", "program tensor list missing value");

    fixture = ProgramFixture();
    fixture.sym_int_values = "{\"size\":{\"as_expr\":{\"expr_str\":\"Symbol('u0', integer=True)\",\"hint\":null}},\"static_size\":{\"as_int\":2}}";
    fixture.sym_bool_values = "{\"guard\":{\"as_expr\":{\"expr_str\":\"GreaterThan(Symbol('u0', integer=True), Integer(0))\",\"hint\":{\"as_bool\":true}}},\"static_guard\":{\"as_bool\":true}}";
    fixture.sym_float_values = "{\"scale\":{\"as_expr\":{\"expr_str\":\"FloatTrueDiv(Symbol('u0', integer=True), Integer(2))\",\"hint\":{\"as_float\":1.5}}},\"static_scale\":{\"as_float\":1.0}}";
    fixture.range_constraints = "{\"u0\":{\"min_val\":0,\"max_val\":48},\"u1\":{\"min_val\":null,\"max_val\":null}}";
    pnnx::ExportedProgram symbolic_program;
    if (parse_program_fixture(fixture, symbolic_program, "program symbolic values"))
    {
        check(symbolic_program.graph.sym_int_values["size"].type == pnnx::EXPORTED_SYM_INT_EXPRESSION && symbolic_program.graph.sym_int_values["size"].expression == "Symbol('u0', integer=True)", "program symbolic values", "symbolic int changed");
        check(symbolic_program.graph.sym_bool_values["guard"].is_expression && symbolic_program.graph.sym_bool_values["guard"].has_hint && symbolic_program.graph.sym_bool_values["guard"].hint, "program symbolic values", "symbolic bool changed");
        check(symbolic_program.graph.sym_float_values["scale"].is_expression && symbolic_program.graph.sym_float_values["scale"].has_hint && symbolic_program.graph.sym_float_values["scale"].hint == 1.5, "program symbolic values", "symbolic float changed");
        check(symbolic_program.range_constraints["u0"].has_min && symbolic_program.range_constraints["u0"].min == 0 && symbolic_program.range_constraints["u0"].has_max && symbolic_program.range_constraints["u0"].max == 48, "program symbolic values", "range constraint changed");
        check(!symbolic_program.range_constraints["u1"].has_min && !symbolic_program.range_constraints["u1"].has_max, "program symbolic values", "unbounded range changed");
    }

    fixture = ProgramFixture();
    fixture.sym_int_values = "{\"s0\":{\"as_expr\":{\"hint\":null}}}";
    expect_program_error(fixture, "$.graph_module.graph.sym_int_values.s0.as_expr.expr_str", "missing required field", "program symbolic expression missing text");

    fixture = ProgramFixture();
    fixture.range_constraints = "{\"s0\":{\"min_val\":0}}";
    expect_program_error(fixture, "$.range_constraints.s0.max_val", "missing required field", "program range missing maximum");

    fixture = ProgramFixture();
    fixture.is_single_tensor_return = "true";
    expect_program_error(fixture, "$.graph_module.graph.is_single_tensor_return", "higher-order single tensor graph is unsupported", "program single tensor return");

    fixture = ProgramFixture();
    fixture.module_call_graph = "{}";
    expect_program_error(fixture, "$.graph_module.module_call_graph", "expected array", "program module call graph type");

    fixture = ProgramFixture();
    fixture.input_specs = "[{\"future_input\":{}}]";
    expect_program_error(fixture, "$.graph_module.signature.input_specs[0]", "unknown input spec tag future_input", "program unknown input spec");

    fixture = ProgramFixture();
    fixture.input_specs = "[{\"user_input\":{\"arg\":" + tensor_argument_json("x") + "},\"token\":{\"arg\":{\"name\":\"t\"}}}]";
    expect_program_error(fixture, "$.graph_module.signature.input_specs[0]", "input spec union must contain exactly one tag", "program multiple input spec tags");

    fixture = ProgramFixture();
    fixture.output_specs = "[{\"future_output\":{}}]";
    expect_program_error(fixture, "$.graph_module.signature.output_specs[0]", "unknown output spec tag future_output", "program unknown output spec");

    fixture = ProgramFixture();
    fixture.output_specs = "[{\"user_output\":{\"arg\":" + tensor_argument_json("y") + "},\"loss_output\":{\"arg\":{\"name\":\"y\"}}}]";
    expect_program_error(fixture, "$.graph_module.signature.output_specs[0]", "output spec union must contain exactly one tag", "program multiple output spec tags");

    fixture = ProgramFixture();
    fixture.input_specs = "[]";
    expect_program_error(fixture, "$.graph_module.signature.input_specs", "input spec count does not match graph inputs", "program input spec count");

    fixture = ProgramFixture();
    fixture.input_specs = "[{\"user_input\":{\"arg\":" + tensor_argument_json("signature_missing") + "}}]";
    expect_program_error(fixture, "$.graph_module.graph.tensor_values.signature_missing", "missing tensor metadata", "program signature missing tensor value");

    fixture = ProgramFixture();
    fixture.output_specs = "[]";
    expect_program_error(fixture, "$.graph_module.signature.output_specs", "output spec count does not match graph outputs", "program output spec count");

    fixture = ProgramFixture();
    fixture.schema_minor = "14";
    fixture.torch_version = "2.9.0";
    fixture.nodes = "[{\"target\":\"torch.ops.aten.relu.default\",\"inputs\":[],\"outputs\":[" + tensor_argument_json("y") + "],\"metadata\":{},\"is_hop_single_tensor_return\":null}]";
    pnnx::ExportedProgram program;
    if (parse_program_fixture(fixture, program, "program schema 8.14 node without name"))
        check(program.graph.nodes.size() == 1 && !program.graph.nodes[0].has_name && program.graph.nodes[0].name.empty(), "program schema 8.14 node without name", "missing node name changed");
}

static std::string join_symints(const std::vector<pnnx::ExportedSymInt>& values)
{
    std::ostringstream text;
    for (size_t i = 0; i < values.size(); i++)
    {
        if (i != 0)
            text << ',';
        if (values[i].type == pnnx::EXPORTED_SYM_INT_STATIC)
            text << values[i].value;
        else
            text << values[i].expression;
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
                join_symints(meta.sizes).c_str(),
                join_symints(meta.strides).c_str(),
                (long long)meta.storage_offset.value,
                meta.device_type.c_str());
        if (meta.has_device_index)
            fprintf(stdout, "%lld", (long long)meta.device_index);
        else
            fprintf(stdout, "null");
        fprintf(stdout, "|%lld|%d\n", (long long)meta.layout, meta.requires_grad ? 1 : 0);
    }
}

static const char* input_kind_name(pnnx::ExportedInputKind kind)
{
    if (kind == pnnx::EXPORTED_USER_INPUT)
        return "user_input";
    if (kind == pnnx::EXPORTED_PARAMETER)
        return "parameter";
    if (kind == pnnx::EXPORTED_BUFFER)
        return "buffer";
    if (kind == pnnx::EXPORTED_TENSOR_CONSTANT)
        return "tensor_constant";
    if (kind == pnnx::EXPORTED_CONSTANT_INPUT)
        return "constant_input";
    if (kind == pnnx::EXPORTED_CUSTOM_OBJ)
        return "custom_obj";
    return "token";
}

static const char* output_kind_name(pnnx::ExportedOutputKind kind)
{
    if (kind == pnnx::EXPORTED_USER_OUTPUT)
        return "user_output";
    if (kind == pnnx::EXPORTED_LOSS_OUTPUT)
        return "loss_output";
    if (kind == pnnx::EXPORTED_BUFFER_MUTATION)
        return "buffer_mutation";
    if (kind == pnnx::EXPORTED_PARAMETER_MUTATION)
        return "parameter_mutation";
    if (kind == pnnx::EXPORTED_GRADIENT_TO_PARAMETER)
        return "gradient_to_parameter";
    if (kind == pnnx::EXPORTED_GRADIENT_TO_USER_INPUT)
        return "gradient_to_user_input";
    if (kind == pnnx::EXPORTED_USER_INPUT_MUTATION)
        return "user_input_mutation";
    return "token";
}

static void print_graph(const pnnx::ExportedProgram& program)
{
    fprintf(stdout, "graph|%llu|%llu|%llu|%llu|%llu|%llu|%llu\n",
            (unsigned long long)program.graph.inputs.size(),
            (unsigned long long)program.graph.nodes.size(),
            (unsigned long long)program.graph.outputs.size(),
            (unsigned long long)program.graph.tensor_values.size(),
            (unsigned long long)program.graph.custom_obj_values.size(),
            (unsigned long long)program.input_specs.size(),
            (unsigned long long)program.output_specs.size());

    for (size_t i = 0; i < program.input_specs.size(); i++)
    {
        const pnnx::ExportedInputSpec& spec = program.input_specs[i];
        fprintf(stdout, "input|%s|%s|%s|%d\n", input_kind_name(spec.kind), spec.arg.name.c_str(), spec.target.c_str(), spec.persistent ? 1 : 0);
    }
    for (size_t i = 0; i < program.graph.nodes.size(); i++)
    {
        const pnnx::ExportedNode& node = program.graph.nodes[i];
        fprintf(stdout, "node|%s|%s|%llu|%llu\n", node.name.c_str(), node.target.c_str(), (unsigned long long)node.inputs.size(), (unsigned long long)node.outputs.size());
    }
    for (size_t i = 0; i < program.output_specs.size(); i++)
    {
        const pnnx::ExportedOutputSpec& spec = program.output_specs[i];
        fprintf(stdout, "output|%s|%s|%s\n", output_kind_name(spec.kind), spec.arg.name.c_str(), spec.target.c_str());
    }
}

static int inspect_package(const std::string& path, bool print_graph_details)
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

    pnnx::ExportedProgram program;
    pnnx::ExportedPayloadConfig weights;
    pnnx::ExportedPayloadConfig constants;
    pnnx::ExportedSchemaError schema_error;
    if (pnnx::parse_exported_program(model_json, program, schema_error) != 0
        || pnnx::parse_exported_payload_config(weights_json, weights, schema_error) != 0
        || pnnx::parse_exported_payload_config(constants_json, constants, schema_error) != 0)
    {
        fprintf(stderr, "%s %s\n", schema_error.path.c_str(), schema_error.message.c_str());
        return 1;
    }

    fprintf(stdout, "header|%d|%d|%s|%lld|%llu|%llu\n",
            program.header.schema_major,
            program.header.schema_minor,
            program.header.torch_version.c_str(),
            (long long)program.header.opset_version.find("aten")->second,
            (unsigned long long)weights.entries.size(),
            (unsigned long long)constants.entries.size());

    if (print_graph_details)
        print_graph(program);

    print_payload_config("weights", weights);
    print_payload_config("constants", constants);

    return 0;
}

int main(int argc, char** argv)
{
    if (argc == 3 && std::string(argv[1]) == "inspect")
        return inspect_package(argv[2], false);
    if (argc == 3 && std::string(argv[1]) == "inspect-graph")
        return inspect_package(argv[2], true);
    if (argc != 1)
        return 2;

    test_headers();
    test_tensor_meta();
    test_payload_config();
    test_tiny_program();
    test_graph_arguments();
    test_graph_signatures();
    test_output_tree_specs();
    test_program_errors();

    if (test_failures != 0)
    {
        fprintf(stderr, "%d exported program schema tests failed across %d paths\n", test_failures, test_paths);
        return 1;
    }

    fprintf(stdout, "exported program schema tests passed %d paths\n", test_paths);
    return 0;
}
