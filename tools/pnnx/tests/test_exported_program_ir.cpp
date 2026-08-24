// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "load_exported_program.h"
#include "pass_level2.h"
#include "pt2_archive.h"

#include <limits.h>
#include <stdio.h>

#include <map>
#include <set>
#include <string>
#include <vector>

static int test_failures = 0;
static int test_paths = 0;

static void check(bool condition, const char* name, const std::string& detail)
{
    test_paths++;
    if (condition)
        return;

    fprintf(stderr, "FAIL %s: %s\n", name, detail.c_str());
    test_failures++;
}

static pnnx::ExportedArgument make_tensor(const std::string& name)
{
    pnnx::ExportedArgument value;
    value.type = pnnx::EXPORTED_ARGUMENT_TENSOR;
    value.name = name;
    return value;
}

static pnnx::ExportedArgument make_tensors(const std::vector<std::string>& names)
{
    pnnx::ExportedArgument value;
    value.type = pnnx::EXPORTED_ARGUMENT_TENSOR_LIST;
    value.tensor_names = names;
    return value;
}

static pnnx::ExportedNamedArgument make_input(const std::string& name, const pnnx::ExportedArgument& value)
{
    pnnx::ExportedNamedArgument input;
    input.name = name;
    input.arg = value;
    input.kind = pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL;
    return input;
}

static pnnx::ExportedNamedArgument make_keyword_input(const std::string& name, const pnnx::ExportedArgument& value)
{
    pnnx::ExportedNamedArgument input = make_input(name, value);
    input.kind = pnnx::EXPORTED_ARGUMENT_KIND_KEYWORD;
    return input;
}

static pnnx::ExportedArgument make_bool(bool value)
{
    pnnx::ExportedArgument argument;
    argument.type = pnnx::EXPORTED_ARGUMENT_BOOL;
    argument.bool_value = value;
    return argument;
}

static pnnx::ExportedArgument make_float(double value)
{
    pnnx::ExportedArgument argument;
    argument.type = pnnx::EXPORTED_ARGUMENT_FLOAT;
    argument.float_value = value;
    return argument;
}

static pnnx::ExportedArgument make_int(int64_t value)
{
    pnnx::ExportedArgument argument;
    argument.type = pnnx::EXPORTED_ARGUMENT_INT;
    argument.int_value = value;
    return argument;
}

static pnnx::ExportedArgument make_ints(const std::vector<int64_t>& values)
{
    pnnx::ExportedArgument argument;
    argument.type = pnnx::EXPORTED_ARGUMENT_INT_LIST;
    argument.int_values = values;
    return argument;
}

static pnnx::ExportedArgument make_string(const std::string& value)
{
    pnnx::ExportedArgument argument;
    argument.type = pnnx::EXPORTED_ARGUMENT_STRING;
    argument.string_value = value;
    return argument;
}

static pnnx::ExportedArgument make_enum(pnnx::ExportedArgumentType type, int64_t value)
{
    pnnx::ExportedArgument argument;
    argument.type = type;
    argument.enum_value = value;
    return argument;
}

static pnnx::ExportedNamedArgument make_input(const std::string& name, const std::string& tensor_name)
{
    pnnx::ExportedNamedArgument input;
    input.name = name;
    input.arg = make_tensor(tensor_name);
    input.kind = pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL;
    return input;
}

static pnnx::ExportedTensorMeta make_tensor_meta(const std::vector<int64_t>& sizes)
{
    pnnx::ExportedTensorMeta meta;
    meta.dtype = 7;
    meta.sizes = sizes;
    meta.strides.resize(sizes.size());
    int64_t stride = 1;
    for (size_t reverse_i = sizes.size(); reverse_i > 0; reverse_i--)
    {
        const size_t i = reverse_i - 1;
        meta.strides[i] = stride;
        stride *= sizes[i];
    }
    meta.layout = 7;
    meta.device_type = "cpu";
    return meta;
}

static pnnx::MaterializedExportedTensor make_state_tensor(const std::vector<int>& shape, size_t byte_count)
{
    pnnx::MaterializedExportedTensor tensor;
    tensor.pnnx_type = 1;
    tensor.shape = shape;
    tensor.data.resize(byte_count, 0);
    return tensor;
}

static std::map<std::string, pnnx::MaterializedExportedTensor> make_linear_state()
{
    std::map<std::string, pnnx::MaterializedExportedTensor> state;
    state["linear.weight"] = make_state_tensor(std::vector<int>{3, 4}, 48);
    state["linear.bias"] = make_state_tensor(std::vector<int>{3}, 12);
    return state;
}

static pnnx::ExportedProgram make_linear_relu_program()
{
    pnnx::ExportedProgram program;
    program.header.schema_major = 8;
    program.header.schema_minor = 20;
    program.header.torch_version = "2.12.1+cu126";
    program.header.opset_version["aten"] = 10;

    program.graph.inputs.push_back(make_tensor("p_linear_weight"));
    program.graph.inputs.push_back(make_tensor("p_linear_bias"));
    program.graph.inputs.push_back(make_tensor("x"));

    pnnx::ExportedNode linear;
    linear.name = "linear";
    linear.has_name = true;
    linear.target = "torch.ops.aten.linear.default";
    linear.inputs.push_back(make_input("input", "x"));
    linear.inputs.push_back(make_input("weight", "p_linear_weight"));
    linear.inputs.push_back(make_input("bias", "p_linear_bias"));
    linear.outputs.push_back(make_tensor("linear"));
    program.graph.nodes.push_back(linear);

    pnnx::ExportedNode relu;
    relu.name = "relu";
    relu.has_name = true;
    relu.target = "torch.ops.aten.relu.default";
    relu.inputs.push_back(make_input("self", "linear"));
    relu.outputs.push_back(make_tensor("relu"));
    program.graph.nodes.push_back(relu);

    program.graph.outputs.push_back(make_tensor("relu"));

    program.graph.tensor_values["p_linear_weight"] = make_tensor_meta(std::vector<int64_t>{3, 4});
    program.graph.tensor_values["p_linear_bias"] = make_tensor_meta(std::vector<int64_t>{3});
    program.graph.tensor_values["x"] = make_tensor_meta(std::vector<int64_t>{2, 4});
    program.graph.tensor_values["linear"] = make_tensor_meta(std::vector<int64_t>{2, 3});
    program.graph.tensor_values["relu"] = make_tensor_meta(std::vector<int64_t>{2, 3});

    pnnx::ExportedInputSpec weight;
    weight.kind = pnnx::EXPORTED_PARAMETER;
    weight.arg = make_tensor("p_linear_weight");
    weight.target = "linear.weight";
    program.input_specs.push_back(weight);

    pnnx::ExportedInputSpec bias;
    bias.kind = pnnx::EXPORTED_PARAMETER;
    bias.arg = make_tensor("p_linear_bias");
    bias.target = "linear.bias";
    program.input_specs.push_back(bias);

    pnnx::ExportedInputSpec input;
    input.kind = pnnx::EXPORTED_USER_INPUT;
    input.arg = make_tensor("x");
    program.input_specs.push_back(input);

    pnnx::ExportedOutputSpec output;
    output.kind = pnnx::EXPORTED_USER_OUTPUT;
    output.arg = make_tensor("relu");
    program.output_specs.push_back(output);

    return program;
}

static pnnx::ExportedProgram make_conv2d_program()
{
    pnnx::ExportedProgram program;
    program.header.schema_major = 8;
    program.header.schema_minor = 20;
    program.header.torch_version = "2.12.1+cu126";
    program.header.opset_version["aten"] = 10;

    program.graph.inputs.push_back(make_tensor("p_conv_weight"));
    program.graph.inputs.push_back(make_tensor("x"));

    pnnx::ExportedNode conv;
    conv.name = "conv2d";
    conv.has_name = true;
    conv.target = "torch.ops.aten.conv2d.default";
    conv.inputs.push_back(make_input("input", make_tensor("x")));
    conv.inputs.push_back(make_input("weight", make_tensor("p_conv_weight")));
    conv.outputs.push_back(make_tensor("conv2d"));
    program.graph.nodes.push_back(conv);
    program.graph.outputs.push_back(make_tensor("conv2d"));

    program.graph.tensor_values["p_conv_weight"] = make_tensor_meta(std::vector<int64_t>{3, 4, 3, 3});
    program.graph.tensor_values["x"] = make_tensor_meta(std::vector<int64_t>{1, 4, 8, 8});
    program.graph.tensor_values["conv2d"] = make_tensor_meta(std::vector<int64_t>{1, 3, 6, 6});

    pnnx::ExportedInputSpec weight;
    weight.kind = pnnx::EXPORTED_PARAMETER;
    weight.arg = make_tensor("p_conv_weight");
    weight.target = "conv.weight";
    program.input_specs.push_back(weight);

    pnnx::ExportedInputSpec input;
    input.kind = pnnx::EXPORTED_USER_INPUT;
    input.arg = make_tensor("x");
    program.input_specs.push_back(input);

    pnnx::ExportedOutputSpec output;
    output.kind = pnnx::EXPORTED_USER_OUTPUT;
    output.arg = make_tensor("conv2d");
    program.output_specs.push_back(output);

    return program;
}

static pnnx::ExportedProgram make_batch_norm_program()
{
    pnnx::ExportedProgram program;
    program.header.schema_major = 8;
    program.header.schema_minor = 20;
    program.header.torch_version = "2.12.1+cu126";
    program.header.opset_version["aten"] = 10;

    const char* state_names[] = {"p_weight", "p_bias", "b_running_mean", "b_running_var"};
    for (size_t i = 0; i < sizeof(state_names) / sizeof(state_names[0]); i++)
        program.graph.inputs.push_back(make_tensor(state_names[i]));
    program.graph.inputs.push_back(make_tensor("x"));

    pnnx::ExportedNode batch_norm;
    batch_norm.name = "batch_norm";
    batch_norm.has_name = true;
    batch_norm.target = "torch.ops.aten.batch_norm.default";
    batch_norm.inputs.push_back(make_input("input", make_tensor("x")));
    batch_norm.inputs.push_back(make_input("weight", make_tensor("p_weight")));
    batch_norm.inputs.push_back(make_input("bias", make_tensor("p_bias")));
    batch_norm.inputs.push_back(make_input("running_mean", make_tensor("b_running_mean")));
    batch_norm.inputs.push_back(make_input("running_var", make_tensor("b_running_var")));
    batch_norm.inputs.push_back(make_input("training", make_bool(false)));
    batch_norm.inputs.push_back(make_input("momentum", make_float(0.1)));
    batch_norm.inputs.push_back(make_input("eps", make_float(1e-5)));
    batch_norm.inputs.push_back(make_input("cudnn_enabled", make_bool(false)));
    batch_norm.outputs.push_back(make_tensor("batch_norm"));
    program.graph.nodes.push_back(batch_norm);
    program.graph.outputs.push_back(make_tensor("batch_norm"));

    program.graph.tensor_values["p_weight"] = make_tensor_meta(std::vector<int64_t>{3});
    program.graph.tensor_values["p_bias"] = make_tensor_meta(std::vector<int64_t>{3});
    program.graph.tensor_values["b_running_mean"] = make_tensor_meta(std::vector<int64_t>{3});
    program.graph.tensor_values["b_running_var"] = make_tensor_meta(std::vector<int64_t>{3});
    program.graph.tensor_values["x"] = make_tensor_meta(std::vector<int64_t>{1, 3, 6, 6});
    program.graph.tensor_values["batch_norm"] = make_tensor_meta(std::vector<int64_t>{1, 3, 6, 6});

    const pnnx::ExportedInputKind state_kinds[] = {pnnx::EXPORTED_PARAMETER, pnnx::EXPORTED_PARAMETER, pnnx::EXPORTED_BUFFER, pnnx::EXPORTED_BUFFER};
    const char* state_targets[] = {"weight", "bias", "running_mean", "running_var"};
    for (size_t i = 0; i < sizeof(state_names) / sizeof(state_names[0]); i++)
    {
        pnnx::ExportedInputSpec state;
        state.kind = state_kinds[i];
        state.arg = make_tensor(state_names[i]);
        state.target = state_targets[i];
        state.persistent = true;
        program.input_specs.push_back(state);
    }

    pnnx::ExportedInputSpec input;
    input.kind = pnnx::EXPORTED_USER_INPUT;
    input.arg = make_tensor("x");
    program.input_specs.push_back(input);

    pnnx::ExportedOutputSpec output;
    output.kind = pnnx::EXPORTED_USER_OUTPUT;
    output.arg = make_tensor("batch_norm");
    program.output_specs.push_back(output);

    return program;
}

static pnnx::ExportedProgram make_inplace_relu_program()
{
    pnnx::ExportedProgram program;
    program.header.schema_major = 8;
    program.header.schema_minor = 20;
    program.header.torch_version = "2.12.1+cu126";
    program.header.opset_version["aten"] = 10;
    program.graph.inputs.push_back(make_tensor("x"));

    pnnx::ExportedNode relu;
    relu.name = "relu_";
    relu.has_name = true;
    relu.target = "torch.ops.aten.relu_.default";
    relu.inputs.push_back(make_input("self", make_tensor("x")));
    relu.outputs.push_back(make_tensor("relu_"));
    program.graph.nodes.push_back(relu);
    program.graph.outputs.push_back(make_tensor("relu_"));

    program.graph.tensor_values["x"] = make_tensor_meta(std::vector<int64_t>{2, 3});
    program.graph.tensor_values["relu_"] = make_tensor_meta(std::vector<int64_t>{2, 3});

    pnnx::ExportedInputSpec input;
    input.kind = pnnx::EXPORTED_USER_INPUT;
    input.arg = make_tensor("x");
    program.input_specs.push_back(input);

    pnnx::ExportedOutputSpec output;
    output.kind = pnnx::EXPORTED_USER_OUTPUT;
    output.arg = make_tensor("relu_");
    program.output_specs.push_back(output);
    return program;
}

static pnnx::ExportedProgram make_max_pool2d_program()
{
    pnnx::ExportedProgram program;
    program.header.schema_major = 8;
    program.header.schema_minor = 20;
    program.header.torch_version = "2.12.1+cu126";
    program.header.opset_version["aten"] = 10;
    program.graph.inputs.push_back(make_tensor("x"));

    pnnx::ExportedNode max_pool2d;
    max_pool2d.name = "max_pool2d";
    max_pool2d.has_name = true;
    max_pool2d.target = "torch.ops.aten.max_pool2d.default";
    max_pool2d.inputs.push_back(make_input("self", make_tensor("x")));
    max_pool2d.inputs.push_back(make_input("kernel_size", make_ints(std::vector<int64_t>{3, 3})));
    max_pool2d.inputs.push_back(make_input("stride", make_ints(std::vector<int64_t>{2, 2})));
    max_pool2d.inputs.push_back(make_input("padding", make_ints(std::vector<int64_t>{1, 1})));
    max_pool2d.inputs.push_back(make_input("dilation", make_ints(std::vector<int64_t>{1, 1})));
    max_pool2d.inputs.push_back(make_input("ceil_mode", make_bool(false)));
    max_pool2d.outputs.push_back(make_tensor("max_pool2d"));
    program.graph.nodes.push_back(max_pool2d);
    program.graph.outputs.push_back(make_tensor("max_pool2d"));

    program.graph.tensor_values["x"] = make_tensor_meta(std::vector<int64_t>{1, 3, 8, 8});
    program.graph.tensor_values["max_pool2d"] = make_tensor_meta(std::vector<int64_t>{1, 3, 4, 4});

    pnnx::ExportedInputSpec input;
    input.kind = pnnx::EXPORTED_USER_INPUT;
    input.arg = make_tensor("x");
    program.input_specs.push_back(input);

    pnnx::ExportedOutputSpec output;
    output.kind = pnnx::EXPORTED_USER_OUTPUT;
    output.arg = make_tensor("max_pool2d");
    program.output_specs.push_back(output);
    return program;
}

static pnnx::ExportedProgram make_inplace_add_program()
{
    pnnx::ExportedProgram program;
    program.header.schema_major = 8;
    program.header.schema_minor = 20;
    program.header.torch_version = "2.12.1+cu126";
    program.header.opset_version["aten"] = 10;
    program.graph.inputs.push_back(make_tensor("x"));
    program.graph.inputs.push_back(make_tensor("other"));

    pnnx::ExportedNode add;
    add.name = "add_";
    add.has_name = true;
    add.target = "torch.ops.aten.add_.Tensor";
    add.inputs.push_back(make_input("self", make_tensor("x")));
    add.inputs.push_back(make_input("other", make_tensor("other")));
    add.outputs.push_back(make_tensor("add_"));
    program.graph.nodes.push_back(add);
    program.graph.outputs.push_back(make_tensor("add_"));

    program.graph.tensor_values["x"] = make_tensor_meta(std::vector<int64_t>{1, 3, 4, 4});
    program.graph.tensor_values["other"] = make_tensor_meta(std::vector<int64_t>{1, 3, 4, 4});
    program.graph.tensor_values["add_"] = make_tensor_meta(std::vector<int64_t>{1, 3, 4, 4});

    const char* input_names[] = {"x", "other"};
    for (size_t i = 0; i < sizeof(input_names) / sizeof(input_names[0]); i++)
    {
        pnnx::ExportedInputSpec input;
        input.kind = pnnx::EXPORTED_USER_INPUT;
        input.arg = make_tensor(input_names[i]);
        program.input_specs.push_back(input);
    }

    pnnx::ExportedOutputSpec output;
    output.kind = pnnx::EXPORTED_USER_OUTPUT;
    output.arg = make_tensor("add_");
    program.output_specs.push_back(output);
    return program;
}

static pnnx::ExportedProgram make_adaptive_avg_pool2d_program()
{
    pnnx::ExportedProgram program;
    program.header.schema_major = 8;
    program.header.schema_minor = 20;
    program.header.torch_version = "2.12.1+cu126";
    program.header.opset_version["aten"] = 10;
    program.graph.inputs.push_back(make_tensor("x"));

    pnnx::ExportedNode adaptive_avg_pool2d;
    adaptive_avg_pool2d.name = "adaptive_avg_pool2d";
    adaptive_avg_pool2d.has_name = true;
    adaptive_avg_pool2d.target = "torch.ops.aten.adaptive_avg_pool2d.default";
    adaptive_avg_pool2d.inputs.push_back(make_input("self", make_tensor("x")));
    adaptive_avg_pool2d.inputs.push_back(make_input("output_size", make_ints(std::vector<int64_t>{1, 1})));
    adaptive_avg_pool2d.outputs.push_back(make_tensor("adaptive_avg_pool2d"));
    program.graph.nodes.push_back(adaptive_avg_pool2d);
    program.graph.outputs.push_back(make_tensor("adaptive_avg_pool2d"));

    program.graph.tensor_values["x"] = make_tensor_meta(std::vector<int64_t>{1, 512, 7, 7});
    program.graph.tensor_values["adaptive_avg_pool2d"] = make_tensor_meta(std::vector<int64_t>{1, 512, 1, 1});

    pnnx::ExportedInputSpec input;
    input.kind = pnnx::EXPORTED_USER_INPUT;
    input.arg = make_tensor("x");
    program.input_specs.push_back(input);

    pnnx::ExportedOutputSpec output;
    output.kind = pnnx::EXPORTED_USER_OUTPUT;
    output.arg = make_tensor("adaptive_avg_pool2d");
    program.output_specs.push_back(output);
    return program;
}

static pnnx::ExportedProgram make_flatten_program()
{
    pnnx::ExportedProgram program;
    program.header.schema_major = 8;
    program.header.schema_minor = 20;
    program.header.torch_version = "2.12.1+cu126";
    program.header.opset_version["aten"] = 10;
    program.graph.inputs.push_back(make_tensor("x"));

    pnnx::ExportedNode flatten;
    flatten.name = "flatten";
    flatten.has_name = true;
    flatten.target = "torch.ops.aten.flatten.using_ints";
    flatten.inputs.push_back(make_input("self", make_tensor("x")));
    flatten.inputs.push_back(make_input("start_dim", make_int(1)));
    flatten.inputs.push_back(make_input("end_dim", make_int(-1)));
    flatten.outputs.push_back(make_tensor("flatten"));
    program.graph.nodes.push_back(flatten);
    program.graph.outputs.push_back(make_tensor("flatten"));

    program.graph.tensor_values["x"] = make_tensor_meta(std::vector<int64_t>{1, 512, 1, 1});
    program.graph.tensor_values["flatten"] = make_tensor_meta(std::vector<int64_t>{1, 512});

    pnnx::ExportedInputSpec input;
    input.kind = pnnx::EXPORTED_USER_INPUT;
    input.arg = make_tensor("x");
    program.input_specs.push_back(input);

    pnnx::ExportedOutputSpec output;
    output.kind = pnnx::EXPORTED_USER_OUTPUT;
    output.arg = make_tensor("flatten");
    program.output_specs.push_back(output);
    return program;
}

static pnnx::ExportedProgram make_cat_program()
{
    pnnx::ExportedProgram program;
    program.header.schema_major = 8;
    program.header.schema_minor = 20;
    program.header.torch_version = "2.12.1+cu126";
    program.header.opset_version["aten"] = 10;
    program.graph.inputs.push_back(make_tensor("x"));
    program.graph.inputs.push_back(make_tensor("y"));

    pnnx::ExportedNode cat;
    cat.name = "cat";
    cat.has_name = true;
    cat.target = "torch.ops.aten.cat.default";
    cat.inputs.push_back(make_input("tensors", make_tensors(std::vector<std::string>{"x", "y"})));
    cat.inputs.push_back(make_input("dim", make_int(1)));
    cat.outputs.push_back(make_tensor("cat"));
    program.graph.nodes.push_back(cat);
    program.graph.outputs.push_back(make_tensor("cat"));

    program.graph.tensor_values["x"] = make_tensor_meta(std::vector<int64_t>{1, 2});
    program.graph.tensor_values["y"] = make_tensor_meta(std::vector<int64_t>{1, 2});
    program.graph.tensor_values["cat"] = make_tensor_meta(std::vector<int64_t>{1, 4});

    const char* input_names[] = {"x", "y"};
    for (size_t i = 0; i < sizeof(input_names) / sizeof(input_names[0]); i++)
    {
        pnnx::ExportedInputSpec input;
        input.kind = pnnx::EXPORTED_USER_INPUT;
        input.arg = make_tensor(input_names[i]);
        program.input_specs.push_back(input);
    }

    pnnx::ExportedOutputSpec output;
    output.kind = pnnx::EXPORTED_USER_OUTPUT;
    output.arg = make_tensor("cat");
    program.output_specs.push_back(output);
    return program;
}

static pnnx::ExportedProgram make_unary_program(const std::string& target, const std::string& output_name, const std::vector<pnnx::ExportedNamedArgument>& arguments)
{
    pnnx::ExportedProgram program;
    program.header.schema_major = 8;
    program.header.schema_minor = 20;
    program.header.torch_version = "2.12.1+cu126";
    program.header.opset_version["aten"] = 10;
    program.graph.inputs.push_back(make_tensor("x"));

    pnnx::ExportedNode node;
    node.name = output_name;
    node.has_name = true;
    node.target = target;
    node.inputs.push_back(make_input("self", make_tensor("x")));
    node.inputs.insert(node.inputs.end(), arguments.begin(), arguments.end());
    node.outputs.push_back(make_tensor(output_name));
    program.graph.nodes.push_back(node);
    program.graph.outputs.push_back(make_tensor(output_name));

    program.graph.tensor_values["x"] = make_tensor_meta(std::vector<int64_t>{1, 3, 4, 4});
    program.graph.tensor_values[output_name] = make_tensor_meta(std::vector<int64_t>{1, 3, 4, 4});

    pnnx::ExportedInputSpec input;
    input.kind = pnnx::EXPORTED_USER_INPUT;
    input.arg = make_tensor("x");
    program.input_specs.push_back(input);

    pnnx::ExportedOutputSpec output;
    output.kind = pnnx::EXPORTED_USER_OUTPUT;
    output.arg = make_tensor(output_name);
    program.output_specs.push_back(output);
    return program;
}

static pnnx::Operator* find_operator(pnnx::Graph& graph, const std::string& type)
{
    for (size_t i = 0; i < graph.ops.size(); i++)
    {
        if (graph.ops[i]->type == type)
            return graph.ops[i];
    }
    return 0;
}

static int count_operator(const pnnx::Graph& graph, const std::string& type)
{
    int count = 0;
    for (size_t i = 0; i < graph.ops.size(); i++)
    {
        if (graph.ops[i]->type == type)
            count++;
    }
    return count;
}

static void test_linear_relu_graph()
{
    const pnnx::ExportedProgram program = make_linear_relu_program();
    std::map<std::string, pnnx::MaterializedExportedTensor> state;
    state["linear.weight"] = make_state_tensor(std::vector<int>{3, 4}, 48);
    state["linear.bias"] = make_state_tensor(std::vector<int>{3}, 12);

    pnnx::Graph graph;
    std::string error = "stale";
    const int result = pnnx::lower_exported_program(program, state, graph, error);

    check(result == 0, "linear relu graph", "lowering failed: " + error);
    check(error.empty(), "linear relu graph", "success retained an error");
    if (result != 0)
        return;

    check(graph.ops.size() == 6, "linear relu graph", "wrong operator count");
    check(graph.operands.size() == 5, "linear relu graph", "wrong operand count");
    check(count_operator(graph, "pnnx.Input") == 1, "linear relu graph", "wrong input count");
    check(count_operator(graph, "pnnx.Attribute") == 2, "linear relu graph", "wrong attribute count");
    check(graph.ops.size() >= 6 && graph.ops[0]->type == "pnnx.Attribute" && graph.ops[0]->name == "linear.weight", "linear relu graph", "weight attribute is not first");
    check(graph.ops.size() >= 6 && graph.ops[1]->type == "pnnx.Attribute" && graph.ops[1]->name == "linear.bias", "linear relu graph", "bias attribute is not second");
    check(graph.ops.size() >= 6 && graph.ops[2]->type == "pnnx.Input" && graph.ops[2]->name == "pnnx_input_0", "linear relu graph", "user input is not third");
    check(graph.ops.size() >= 6 && graph.ops[3]->type == "aten::linear" && graph.ops[3]->name == "linear", "linear relu graph", "linear node is not fourth");
    check(graph.ops.size() >= 6 && graph.ops[4]->type == "aten::relu" && graph.ops[4]->name == "relu", "linear relu graph", "relu node is not fifth");
    check(graph.ops.size() >= 6 && graph.ops[5]->type == "pnnx.Output" && graph.ops[5]->name == "pnnx_output_0", "linear relu graph", "output is not last");

    pnnx::Operator* linear = find_operator(graph, "aten::linear");
    check(linear != 0, "linear relu graph", "missing aten::linear");
    check(linear && linear->inputnames == std::vector<std::string>({"input", "weight", "bias"}), "linear relu graph", "linear input names are not canonical");
    check(linear && linear->inputs.size() == 3 && linear->inputs[0]->name == "x" && linear->inputs[1]->name == "p_linear_weight" && linear->inputs[2]->name == "p_linear_bias", "linear relu graph", "linear inputs are out of order");

    pnnx::Operand* weight = graph.get_operand("p_linear_weight");
    check(weight != 0, "linear relu graph", "missing weight operand");
    check(weight && weight->producer && weight->producer->type == "pnnx.Attribute", "linear relu graph", "weight is not produced by an attribute");
    check(weight && weight->type == 1 && weight->shape == std::vector<int>({3, 4}), "linear relu graph", "weight metadata is wrong");
    check(weight && weight->producer && weight->producer->has_attr("data") && weight->producer->attrs.at("data").data.size() == 48, "linear relu graph", "weight bytes were not attached");
    check(weight && weight->consumers.size() == 1 && weight->consumers[0] == linear, "linear relu graph", "weight consumer link is wrong");

    pnnx::Operand* relu = graph.get_operand("relu");
    check(relu && relu->producer && relu->producer->type == "aten::relu", "linear relu graph", "relu producer link is wrong");
    check(relu && relu->consumers.size() == 1 && relu->consumers[0]->type == "pnnx.Output", "linear relu graph", "relu output consumer link is wrong");
    check(relu && relu->type == 1 && relu->shape == std::vector<int>({2, 3}), "linear relu graph", "relu metadata is wrong");

    pnnx::pass_level2(graph);
    check(count_operator(graph, "aten::linear") == 0 && count_operator(graph, "F.linear") == 1, "linear relu pass level2", "linear was not canonicalized");
    check(count_operator(graph, "aten::relu") == 0 && count_operator(graph, "F.relu") == 1, "linear relu pass level2", "relu was not canonicalized");
}

static void test_linear_default_bias_constant()
{
    pnnx::ExportedProgram program = make_linear_relu_program();
    program.graph.inputs.erase(program.graph.inputs.begin() + 1);
    program.input_specs.erase(program.input_specs.begin() + 1);
    program.graph.tensor_values.erase("p_linear_bias");
    program.graph.nodes[0].inputs.erase(program.graph.nodes[0].inputs.begin() + 2);

    std::map<std::string, pnnx::MaterializedExportedTensor> state;
    state["linear.weight"] = make_state_tensor(std::vector<int>{3, 4}, 48);

    pnnx::Graph graph;
    std::string error = "stale";
    const int result = pnnx::lower_exported_program(program, state, graph, error);

    check(result == 0, "linear default bias", "lowering failed: " + error);
    check(error.empty(), "linear default bias", "success retained an error");
    if (result != 0)
        return;

    check(count_operator(graph, "prim::Constant") == 1, "linear default bias", "missing None constant");

    pnnx::Operator* linear = find_operator(graph, "aten::linear");
    check(linear && linear->inputs.size() == 3, "linear default bias", "linear does not have three canonical inputs");
    check(linear && linear->inputnames == std::vector<std::string>({"input", "weight", "bias"}), "linear default bias", "linear input names are not canonical");
    check(linear && linear->inputs.size() == 3 && linear->inputs[2]->producer && linear->inputs[2]->producer->type == "prim::Constant", "linear default bias", "bias is not produced by a constant");
    check(linear && linear->inputs.size() == 3 && linear->inputs[2]->producer && linear->inputs[2]->producer->has_param("value") && linear->inputs[2]->producer->params.at("value").type == 0, "linear default bias", "bias constant is not None");

    pnnx::pass_level2(graph);
    check(count_operator(graph, "F.linear") == 1, "linear default bias", "linear with None bias was not canonicalized");
}

static void test_conv2d_constant_arguments()
{
    const pnnx::ExportedProgram program = make_conv2d_program();
    std::map<std::string, pnnx::MaterializedExportedTensor> state;
    state["conv.weight"] = make_state_tensor(std::vector<int>{3, 4, 3, 3}, 432);

    pnnx::Graph graph;
    std::string error = "stale";
    const int result = pnnx::lower_exported_program(program, state, graph, error);

    check(result == 0, "conv2d constants", "lowering failed: " + error);
    check(error.empty(), "conv2d constants", "success retained an error");
    if (result != 0)
        return;

    pnnx::Operator* conv = find_operator(graph, "aten::conv2d");
    check(conv != 0, "conv2d constants", "missing aten::conv2d");
    check(conv && conv->inputnames == std::vector<std::string>({"input", "weight", "bias", "stride", "padding", "dilation", "groups"}), "conv2d constants", "conv2d input names are not canonical");
    check(conv && conv->inputs.size() == 7, "conv2d constants", "conv2d does not have seven inputs");
    if (!conv || conv->inputs.size() != 7)
        return;

    const pnnx::Parameter& bias = conv->inputs[2]->producer->params.at("value");
    const pnnx::Parameter& stride = conv->inputs[3]->producer->params.at("value");
    const pnnx::Parameter& padding = conv->inputs[4]->producer->params.at("value");
    const pnnx::Parameter& dilation = conv->inputs[5]->producer->params.at("value");
    const pnnx::Parameter& groups = conv->inputs[6]->producer->params.at("value");
    check(bias.type == 0, "conv2d constants", "bias default is not None");
    check(stride.type == 5 && stride.ai == std::vector<int>({1, 1}), "conv2d constants", "stride default is wrong");
    check(padding.type == 5 && padding.ai == std::vector<int>({0, 0}), "conv2d constants", "padding default is wrong");
    check(dilation.type == 5 && dilation.ai == std::vector<int>({1, 1}), "conv2d constants", "dilation default is wrong");
    check(groups.type == 2 && groups.i == 1, "conv2d constants", "groups default is wrong");

    pnnx::pass_level2(graph);
    check(count_operator(graph, "aten::conv2d") == 0 && count_operator(graph, "F.conv2d") == 1, "conv2d pass level2", "conv2d was not canonicalized");
}

static void test_batch_norm_constant_arguments()
{
    const pnnx::ExportedProgram program = make_batch_norm_program();
    std::map<std::string, pnnx::MaterializedExportedTensor> state;
    state["weight"] = make_state_tensor(std::vector<int>{3}, 12);
    state["bias"] = make_state_tensor(std::vector<int>{3}, 12);
    state["running_mean"] = make_state_tensor(std::vector<int>{3}, 12);
    state["running_var"] = make_state_tensor(std::vector<int>{3}, 12);

    pnnx::Graph graph;
    std::string error = "stale";
    const int result = pnnx::lower_exported_program(program, state, graph, error);

    check(result == 0, "batch norm constants", "lowering failed: " + error);
    check(error.empty(), "batch norm constants", "success retained an error");
    if (result != 0)
        return;

    pnnx::Operator* batch_norm = find_operator(graph, "aten::batch_norm");
    check(batch_norm != 0, "batch norm constants", "missing aten::batch_norm");
    check(batch_norm && batch_norm->inputnames == std::vector<std::string>({"input", "weight", "bias", "running_mean", "running_var", "training", "momentum", "eps", "cudnn_enabled"}), "batch norm constants", "batch_norm input names are not canonical");
    check(batch_norm && batch_norm->inputs.size() == 9, "batch norm constants", "batch_norm does not have nine inputs");
    if (!batch_norm || batch_norm->inputs.size() != 9)
        return;

    const pnnx::Parameter& training = batch_norm->inputs[5]->producer->params.at("value");
    const pnnx::Parameter& momentum = batch_norm->inputs[6]->producer->params.at("value");
    const pnnx::Parameter& eps = batch_norm->inputs[7]->producer->params.at("value");
    const pnnx::Parameter& cudnn_enabled = batch_norm->inputs[8]->producer->params.at("value");
    check(training.type == 1 && !training.b, "batch norm constants", "training constant is wrong");
    check(momentum.type == 3 && momentum.f == 0.1f, "batch norm constants", "momentum constant is wrong");
    check(eps.type == 3 && eps.f == 1e-5f, "batch norm constants", "eps constant is wrong");
    check(cudnn_enabled.type == 1 && !cudnn_enabled.b, "batch norm constants", "cudnn_enabled constant is wrong");

    pnnx::pass_level2(graph);
    check(count_operator(graph, "aten::batch_norm") == 0 && count_operator(graph, "F.batch_norm") == 1, "batch norm pass level2", "batch_norm was not canonicalized");
}

static void test_inplace_relu_is_functionalized_by_level2()
{
    const pnnx::ExportedProgram program = make_inplace_relu_program();
    pnnx::Graph graph;
    std::string error = "stale";
    const int result = pnnx::lower_exported_program(program, std::map<std::string, pnnx::MaterializedExportedTensor>(), graph, error);

    check(result == 0, "inplace relu", "lowering failed: " + error);
    check(error.empty(), "inplace relu", "success retained an error");
    if (result != 0)
        return;

    check(count_operator(graph, "aten::relu_") == 1 && count_operator(graph, "aten::relu") == 0, "inplace relu", "importer did not preserve the inplace operator");
    pnnx::pass_level2(graph);
    check(count_operator(graph, "aten::relu_") == 0 && count_operator(graph, "F.relu") == 1, "inplace relu pass level2", "relu_ was not functionalized and canonicalized");
}

static void test_max_pool2d_constant_arguments()
{
    const pnnx::ExportedProgram program = make_max_pool2d_program();
    pnnx::Graph graph;
    std::string error = "stale";
    const int result = pnnx::lower_exported_program(program, std::map<std::string, pnnx::MaterializedExportedTensor>(), graph, error);

    check(result == 0, "max pool2d constants", "lowering failed: " + error);
    check(error.empty(), "max pool2d constants", "success retained an error");
    if (result != 0)
        return;

    pnnx::Operator* max_pool2d = find_operator(graph, "aten::max_pool2d");
    check(max_pool2d != 0, "max pool2d constants", "missing aten::max_pool2d");
    check(max_pool2d && max_pool2d->inputnames == std::vector<std::string>({"self", "kernel_size", "stride", "padding", "dilation", "ceil_mode"}), "max pool2d constants", "max_pool2d input names are not canonical");
    check(max_pool2d && max_pool2d->inputs.size() == 6, "max pool2d constants", "max_pool2d does not have six inputs");
    if (!max_pool2d || max_pool2d->inputs.size() != 6)
        return;

    const pnnx::Parameter& kernel_size = max_pool2d->inputs[1]->producer->params.at("value");
    const pnnx::Parameter& stride = max_pool2d->inputs[2]->producer->params.at("value");
    const pnnx::Parameter& padding = max_pool2d->inputs[3]->producer->params.at("value");
    const pnnx::Parameter& dilation = max_pool2d->inputs[4]->producer->params.at("value");
    const pnnx::Parameter& ceil_mode = max_pool2d->inputs[5]->producer->params.at("value");
    check(kernel_size.type == 5 && kernel_size.ai == std::vector<int>({3, 3}), "max pool2d constants", "kernel_size constant is wrong");
    check(stride.type == 5 && stride.ai == std::vector<int>({2, 2}), "max pool2d constants", "stride constant is wrong");
    check(padding.type == 5 && padding.ai == std::vector<int>({1, 1}), "max pool2d constants", "padding constant is wrong");
    check(dilation.type == 5 && dilation.ai == std::vector<int>({1, 1}), "max pool2d constants", "dilation constant is wrong");
    check(ceil_mode.type == 1 && !ceil_mode.b, "max pool2d constants", "ceil_mode constant is wrong");

    pnnx::pass_level2(graph);
    check(count_operator(graph, "aten::max_pool2d") == 0 && count_operator(graph, "F.max_pool2d") == 1, "max pool2d pass level2", "max_pool2d was not canonicalized");
}

static void test_inplace_add_is_functionalized_by_level2()
{
    const pnnx::ExportedProgram program = make_inplace_add_program();
    pnnx::Graph graph;
    std::string error = "stale";
    const int result = pnnx::lower_exported_program(program, std::map<std::string, pnnx::MaterializedExportedTensor>(), graph, error);

    check(result == 0, "inplace add", "lowering failed: " + error);
    check(error.empty(), "inplace add", "success retained an error");
    if (result != 0)
        return;

    check(count_operator(graph, "aten::add_") == 1 && count_operator(graph, "aten::add") == 0, "inplace add", "importer did not preserve the inplace operator");
    pnnx::Operator* add = find_operator(graph, "aten::add_");
    check(add && add->inputnames == std::vector<std::string>({"self", "other", "alpha"}), "inplace add", "add input names are not canonical");
    check(add && add->inputs.size() == 3, "inplace add", "add does not have three inputs");
    if (!add || add->inputs.size() != 3)
        return;

    const pnnx::Parameter& alpha = add->inputs[2]->producer->params.at("value");
    check(alpha.type == 2 && alpha.i == 1, "inplace add", "alpha constant is wrong");

    pnnx::pass_level2(graph);
    check(count_operator(graph, "aten::add_") == 0 && count_operator(graph, "aten::add") == 1, "inplace add pass level2", "add_ was not functionalized");
}

static void test_adaptive_avg_pool2d_output_size()
{
    const pnnx::ExportedProgram program = make_adaptive_avg_pool2d_program();
    pnnx::Graph graph;
    std::string error = "stale";
    const int result = pnnx::lower_exported_program(program, std::map<std::string, pnnx::MaterializedExportedTensor>(), graph, error);

    check(result == 0, "adaptive avg pool2d", "lowering failed: " + error);
    check(error.empty(), "adaptive avg pool2d", "success retained an error");
    if (result != 0)
        return;

    pnnx::Operator* adaptive_avg_pool2d = find_operator(graph, "aten::adaptive_avg_pool2d");
    check(adaptive_avg_pool2d != 0, "adaptive avg pool2d", "missing aten::adaptive_avg_pool2d");
    check(adaptive_avg_pool2d && adaptive_avg_pool2d->inputnames == std::vector<std::string>({"self", "output_size"}), "adaptive avg pool2d", "adaptive_avg_pool2d input names are not canonical");
    check(adaptive_avg_pool2d && adaptive_avg_pool2d->inputs.size() == 2, "adaptive avg pool2d", "adaptive_avg_pool2d does not have two inputs");
    if (!adaptive_avg_pool2d || adaptive_avg_pool2d->inputs.size() != 2)
        return;

    const pnnx::Parameter& output_size = adaptive_avg_pool2d->inputs[1]->producer->params.at("value");
    check(output_size.type == 5 && output_size.ai == std::vector<int>({1, 1}), "adaptive avg pool2d", "output_size constant is wrong");

    pnnx::pass_level2(graph);
    check(count_operator(graph, "aten::adaptive_avg_pool2d") == 0 && count_operator(graph, "F.adaptive_avg_pool2d") == 1, "adaptive avg pool2d pass level2", "adaptive_avg_pool2d was not canonicalized");
}

static void test_flatten_dimensions()
{
    const pnnx::ExportedProgram program = make_flatten_program();
    pnnx::Graph graph;
    std::string error = "stale";
    const int result = pnnx::lower_exported_program(program, std::map<std::string, pnnx::MaterializedExportedTensor>(), graph, error);

    check(result == 0, "flatten dimensions", "lowering failed: " + error);
    check(error.empty(), "flatten dimensions", "success retained an error");
    if (result != 0)
        return;

    pnnx::Operator* flatten = find_operator(graph, "aten::flatten");
    check(flatten != 0, "flatten dimensions", "missing aten::flatten");
    check(flatten && flatten->inputnames == std::vector<std::string>({"self", "start_dim", "end_dim"}), "flatten dimensions", "flatten input names are not canonical");
    check(flatten && flatten->inputs.size() == 3, "flatten dimensions", "flatten does not have three inputs");
    if (!flatten || flatten->inputs.size() != 3)
        return;

    const pnnx::Parameter& start_dim = flatten->inputs[1]->producer->params.at("value");
    const pnnx::Parameter& end_dim = flatten->inputs[2]->producer->params.at("value");
    check(start_dim.type == 2 && start_dim.i == 1, "flatten dimensions", "start_dim constant is wrong");
    check(end_dim.type == 2 && end_dim.i == -1, "flatten dimensions", "end_dim constant is wrong");

    pnnx::pass_level2(graph);
    check(count_operator(graph, "aten::flatten") == 0 && count_operator(graph, "torch.flatten") == 1, "flatten pass level2", "flatten was not canonicalized");
}

static void test_dispatcher_backed_target_lowering()
{
    pnnx::ExportedProgram program = make_linear_relu_program();
    program.graph.nodes[1].target = "torch.ops.aten.sigmoid.default";

    pnnx::Graph graph;
    std::string error;
    const int result = pnnx::lower_exported_program(program, make_linear_state(), graph, error);

    check(result == 0, "dispatcher backed target", error);
    if (result != 0)
        return;

    pnnx::Operator* sigmoid = find_operator(graph, "aten::sigmoid");
    check(sigmoid != 0, "dispatcher backed target", "missing aten::sigmoid");
    check(sigmoid && sigmoid->inputnames == std::vector<std::string>({"self"}), "dispatcher backed target", "sigmoid input names are not canonical");
}

static void test_tensor_list_argument_lowering()
{
    const pnnx::ExportedProgram program = make_cat_program();
    pnnx::Graph graph;
    std::string error;
    const int result = pnnx::lower_exported_program(program, std::map<std::string, pnnx::MaterializedExportedTensor>(), graph, error);

    check(result == 0, "tensor list argument", error);
    if (result != 0)
        return;

    pnnx::Operator* list = find_operator(graph, "prim::ListConstruct");
    pnnx::Operator* cat = find_operator(graph, "aten::cat");
    check(list != 0, "tensor list argument", "missing prim::ListConstruct");
    check(list && list->inputs.size() == 2 && list->inputs[0]->name == "x" && list->inputs[1]->name == "y", "tensor list argument", "tensor list inputs are out of order");
    check(list && list->outputs.size() == 1 && list->outputs[0]->consumers.size() == 1 && list->outputs[0]->consumers[0] == cat, "tensor list argument", "tensor list output is not connected to cat");
    check(cat != 0, "tensor list argument", "missing aten::cat");
    check(cat && cat->inputnames == std::vector<std::string>({"tensors", "dim"}), "tensor list argument", "cat input names are not canonical");
    check(cat && cat->inputs.size() == 2 && cat->inputs[0]->producer == list, "tensor list argument", "cat does not consume the tensor list");

    pnnx::pass_level2(graph);
    check(count_operator(graph, "aten::cat") == 0 && count_operator(graph, "torch.cat") == 1, "tensor list pass level2", "cat was not canonicalized");
}

static void test_string_and_memory_format_argument_lowering()
{
    {
        const pnnx::ExportedProgram program = make_unary_program("torch.ops.aten.contiguous.default", "contiguous", std::vector<pnnx::ExportedNamedArgument>());
        pnnx::Graph graph;
        std::string error;
        const int result = pnnx::lower_exported_program(program, std::map<std::string, pnnx::MaterializedExportedTensor>(), graph, error);

        check(result == 0, "memory format argument", error);
        if (result == 0)
        {
            pnnx::Operator* contiguous = find_operator(graph, "aten::contiguous");
            check(contiguous && contiguous->inputnames == std::vector<std::string>({"self", "memory_format"}), "memory format argument", "contiguous input names are not canonical");
            check(contiguous && contiguous->inputs.size() == 2 && contiguous->inputs[1]->producer && contiguous->inputs[1]->producer->has_param("value"), "memory format argument", "memory format constant is missing");
            if (contiguous && contiguous->inputs.size() == 2 && contiguous->inputs[1]->producer && contiguous->inputs[1]->producer->has_param("value"))
            {
                const pnnx::Parameter& memory_format = contiguous->inputs[1]->producer->params.at("value");
                check(memory_format.type == 2 && memory_format.i == 0, "memory format argument", "contiguous format was not converted to the pnnx/JIT enum value");
            }

            pnnx::pass_level2(graph);
            check(count_operator(graph, "aten::contiguous") == 0, "memory format pass level2", "contiguous was not eliminated");
        }
    }

    {
        const int64_t exported_values[] = {1, 2, 3, 4};
        const int pnnx_values[] = {0, 2, 3, 1};
        for (size_t i = 0; i < sizeof(exported_values) / sizeof(exported_values[0]); i++)
        {
            const pnnx::ExportedProgram program = make_unary_program("torch.ops.aten.contiguous.default", "contiguous", std::vector<pnnx::ExportedNamedArgument>{make_keyword_input("memory_format", make_enum(pnnx::EXPORTED_ARGUMENT_MEMORY_FORMAT, exported_values[i]))});
            pnnx::Graph graph;
            std::string error;
            const int result = pnnx::lower_exported_program(program, std::map<std::string, pnnx::MaterializedExportedTensor>(), graph, error);

            check(result == 0, "memory format enum mapping", error);
            if (result == 0)
            {
                pnnx::Operator* contiguous = find_operator(graph, "aten::contiguous");
                check(contiguous && contiguous->inputs.size() == 2 && contiguous->inputs[1]->producer && contiguous->inputs[1]->producer->has_param("value"), "memory format enum mapping", "memory format constant is missing");
                if (contiguous && contiguous->inputs.size() == 2 && contiguous->inputs[1]->producer && contiguous->inputs[1]->producer->has_param("value"))
                {
                    const pnnx::Parameter& memory_format = contiguous->inputs[1]->producer->params.at("value");
                    check(memory_format.type == 2 && memory_format.i == pnnx_values[i], "memory format enum mapping", "memory format enum value is wrong");
                }
            }
        }
    }

    {
        const pnnx::ExportedProgram program = make_unary_program("torch.ops.aten.gelu.default", "gelu", std::vector<pnnx::ExportedNamedArgument>{make_keyword_input("approximate", make_string("tanh"))});
        pnnx::Graph graph;
        std::string error;
        const int result = pnnx::lower_exported_program(program, std::map<std::string, pnnx::MaterializedExportedTensor>(), graph, error);

        check(result == 0, "string argument", error);
        if (result == 0)
        {
            pnnx::Operator* gelu = find_operator(graph, "aten::gelu");
            check(gelu && gelu->inputs.size() == 2 && gelu->inputs[1]->producer && gelu->inputs[1]->producer->has_param("value"), "string argument", "gelu approximate constant is missing");
            if (gelu && gelu->inputs.size() == 2 && gelu->inputs[1]->producer && gelu->inputs[1]->producer->has_param("value"))
            {
                const pnnx::Parameter& approximate = gelu->inputs[1]->producer->params.at("value");
                check(approximate.type == 4 && approximate.s == "tanh", "string argument", "gelu approximate string is wrong");
            }

            pnnx::pass_level2(graph);
            pnnx::Operator* functional_gelu = find_operator(graph, "F.gelu");
            check(functional_gelu && functional_gelu->has_param("approximate") && functional_gelu->params.at("approximate").s == "tanh", "string pass level2", "gelu string was not preserved by the existing pass");
        }
    }
}

static void expect_lower_error(const pnnx::ExportedProgram& program,
                               const std::map<std::string, pnnx::MaterializedExportedTensor>& state,
                               const std::string& expected_error,
                               const char* name)
{
    pnnx::Graph graph;
    std::string error = "stale";
    const int result = pnnx::lower_exported_program(program, state, graph, error);

    check(result != 0, name, "lowering unexpectedly succeeded");
    check(error.find(expected_error) != std::string::npos, name, "wrong error: " + error);
    check(graph.ops.empty() && graph.operands.empty(), name, "failed lowering modified the destination graph");
}

static void test_reject_unsupported_signature_inputs()
{
    {
        pnnx::ExportedProgram program = make_linear_relu_program();
        pnnx::ExportedArgument scalar;
        scalar.type = pnnx::EXPORTED_ARGUMENT_INT;
        scalar.name = "x";
        scalar.int_value = 2;
        program.graph.inputs[2] = scalar;
        program.input_specs[2].arg = scalar;
        expect_lower_error(program, make_linear_state(), "user input x must be a tensor", "non-tensor user input");
    }

    {
        pnnx::ExportedProgram program = make_linear_relu_program();
        program.input_specs[0].kind = pnnx::EXPORTED_BUFFER;
        program.input_specs[0].persistent = false;
        expect_lower_error(program, make_linear_state(), "non-persistent buffer linear.weight is unsupported", "non-persistent buffer");
    }

    {
        pnnx::ExportedProgram program = make_linear_relu_program();
        program.input_specs[2].kind = pnnx::EXPORTED_CUSTOM_OBJ;
        expect_lower_error(program, make_linear_state(), "custom object input x is unsupported", "custom object input");
    }

    {
        pnnx::ExportedProgram program = make_linear_relu_program();
        program.input_specs[2].kind = pnnx::EXPORTED_TOKEN;
        expect_lower_error(program, make_linear_state(), "token input x is unsupported", "token input");
    }
}

static void test_reject_non_inference_outputs()
{
    struct OutputCase
    {
        pnnx::ExportedOutputKind kind;
        const char* message;
        const char* name;
    };

    const OutputCase cases[] = {
        {pnnx::EXPORTED_LOSS_OUTPUT, "loss output", "loss output"},
        {pnnx::EXPORTED_BUFFER_MUTATION, "buffer mutation", "buffer mutation output"},
        {pnnx::EXPORTED_PARAMETER_MUTATION, "parameter mutation", "parameter mutation output"},
        {pnnx::EXPORTED_GRADIENT_TO_PARAMETER, "gradient to parameter", "parameter gradient output"},
        {pnnx::EXPORTED_GRADIENT_TO_USER_INPUT, "gradient to user input", "user input gradient output"},
        {pnnx::EXPORTED_USER_INPUT_MUTATION, "user input mutation", "user input mutation output"},
        {pnnx::EXPORTED_OUTPUT_TOKEN, "output token", "token output"},
    };

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
    {
        pnnx::ExportedProgram program = make_linear_relu_program();
        program.output_specs[0].kind = cases[i].kind;
        expect_lower_error(program, make_linear_state(), cases[i].message, cases[i].name);
    }
}

static void test_reject_invalid_graph_definitions()
{
    {
        pnnx::ExportedProgram program = make_linear_relu_program();
        program.graph.tensor_values["x"].sizes[0] = -1;
        expect_lower_error(program, make_linear_state(), "tensor x has a negative or symbolic size at dimension 0", "negative tensor size");
    }

    {
        pnnx::ExportedProgram program = make_linear_relu_program();
        std::map<std::string, pnnx::MaterializedExportedTensor> state = make_linear_state();
        state.erase("linear.weight");
        expect_lower_error(program, state, "parameter linear.weight is missing materialized state", "missing parameter state");
    }

    {
        pnnx::ExportedProgram program = make_linear_relu_program();
        program.graph.nodes[1].outputs[0] = make_tensor("linear");
        program.graph.outputs[0] = make_tensor("linear");
        program.output_specs[0].arg = make_tensor("linear");
        expect_lower_error(program, make_linear_state(), "tensor value linear is defined more than once", "duplicate tensor definition");
    }

    {
        pnnx::ExportedProgram program = make_linear_relu_program();
        program.graph.nodes[1].target = "aten::relu";
        expect_lower_error(program, make_linear_state(), "torch 2.12.1+cu126 aten opset 10 target aten::relu", "malformed operator context");
    }

    {
        pnnx::ExportedProgram program = make_cat_program();
        program.graph.nodes[0].inputs[0].arg.tensor_names[1] = "missing";
        expect_lower_error(program, std::map<std::string, pnnx::MaterializedExportedTensor>(), "unknown tensor value missing for tensor-list argument tensors", "unknown tensor-list member");
    }

    {
        pnnx::ExportedProgram program = make_unary_program("torch.ops.aten.contiguous.default", "contiguous", std::vector<pnnx::ExportedNamedArgument>{make_keyword_input("memory_format", make_enum(pnnx::EXPORTED_ARGUMENT_MEMORY_FORMAT, 99))});
        expect_lower_error(program, std::map<std::string, pnnx::MaterializedExportedTensor>(), "unsupported non-tensor argument memory_format", "unknown memory format");
    }
}

static void test_reject_signature_graph_mismatch()
{
    {
        pnnx::ExportedProgram program = make_linear_relu_program();
        program.graph.inputs[2] = make_tensor("y");
        program.graph.tensor_values["y"] = make_tensor_meta(std::vector<int64_t>{2, 4});
        expect_lower_error(program, make_linear_state(), "input spec 2 tensor x does not match graph input y", "input signature mismatch");
    }

    {
        pnnx::ExportedProgram program = make_linear_relu_program();
        program.graph.outputs[0] = make_tensor("linear");
        expect_lower_error(program, make_linear_state(), "output spec 0 tensor relu does not match graph output linear", "output signature mismatch");
    }
}

static void test_reject_state_metadata_mismatch()
{
    {
        pnnx::ExportedProgram program = make_linear_relu_program();
        std::map<std::string, pnnx::MaterializedExportedTensor> state = make_linear_state();
        state["linear.weight"].pnnx_type = 3;
        expect_lower_error(program, state, "parameter linear.weight type does not match tensor metadata", "state type mismatch");
    }

    {
        pnnx::ExportedProgram program = make_linear_relu_program();
        std::map<std::string, pnnx::MaterializedExportedTensor> state = make_linear_state();
        state["linear.weight"].shape = std::vector<int>{4, 3};
        expect_lower_error(program, state, "parameter linear.weight shape does not match tensor metadata", "state shape mismatch");
    }

    {
        pnnx::ExportedProgram program = make_linear_relu_program();
        program.graph.tensor_values.erase("p_linear_weight");
        expect_lower_error(program, make_linear_state(), "missing tensor metadata for p_linear_weight", "missing state metadata");
    }

    {
        pnnx::ExportedProgram program = make_linear_relu_program();
        program.graph.tensor_values["x"].layout = 1;
        expect_lower_error(program, make_linear_state(), "unsupported tensor layout 1 for x", "unsupported input layout");
    }
}

static void test_generated_names_do_not_collide()
{
    {
        pnnx::ExportedProgram program = make_linear_relu_program();
        program.graph.inputs.erase(program.graph.inputs.begin() + 1);
        program.input_specs.erase(program.input_specs.begin() + 1);
        program.graph.tensor_values.erase("p_linear_bias");
        program.graph.nodes[0].inputs.erase(program.graph.nodes[0].inputs.begin() + 2);

        program.graph.inputs[1] = make_tensor("pnnx_1");
        program.input_specs[1].arg = make_tensor("pnnx_1");
        program.graph.nodes[0].inputs[0].arg = make_tensor("pnnx_1");
        program.graph.tensor_values["pnnx_1"] = program.graph.tensor_values["x"];
        program.graph.tensor_values.erase("x");

        std::map<std::string, pnnx::MaterializedExportedTensor> state;
        state["linear.weight"] = make_state_tensor(std::vector<int>{3, 4}, 48);
        pnnx::Graph graph;
        std::string error;
        const int result = pnnx::lower_exported_program(program, state, graph, error);

        check(result == 0, "constant operand name collision", "lowering failed: " + error);
        std::set<std::string> names;
        for (size_t i = 0; i < graph.operands.size(); i++)
            names.insert(graph.operands[i]->name);
        check(names.size() == graph.operands.size(), "constant operand name collision", "generated constant reused a tensor value name");
    }

    {
        pnnx::ExportedProgram program = make_linear_relu_program();
        program.graph.nodes[0].name = "linear.weight";
        pnnx::Graph graph;
        std::string error;
        const int result = pnnx::lower_exported_program(program, make_linear_state(), graph, error);

        check(result == 0, "operator name collision", "lowering failed: " + error);
        std::set<std::string> names;
        for (size_t i = 0; i < graph.ops.size(); i++)
            names.insert(graph.ops[i]->name);
        check(names.size() == graph.ops.size(), "operator name collision", "serialized node reused an operator name");
    }
}

static std::string join_operands(const std::vector<pnnx::Operand*>& operands)
{
    std::string result;
    for (size_t i = 0; i < operands.size(); i++)
    {
        if (i != 0)
            result += ',';
        result += operands[i]->name;
    }
    return result;
}

static int inspect_package(const char* path)
{
    pnnx::Pt2ArchiveReader reader;
    std::string error;
    if (reader.open(path, error) != 0)
    {
        fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }

    pnnx::JsonValue model_json;
    if (reader.read_json(reader.layout().model_json_path, model_json, error) != 0)
    {
        fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }

    pnnx::ExportedProgram program;
    pnnx::ExportedSchemaError schema_error;
    if (pnnx::parse_exported_program(model_json, program, schema_error) != 0)
    {
        fprintf(stderr, "%s %s\n", schema_error.path.c_str(), schema_error.message.c_str());
        return 1;
    }

    std::map<std::string, pnnx::MaterializedExportedTensor> state;
    for (size_t i = 0; i < program.input_specs.size(); i++)
    {
        const pnnx::ExportedInputSpec& spec = program.input_specs[i];
        if (spec.kind != pnnx::EXPORTED_PARAMETER && spec.kind != pnnx::EXPORTED_BUFFER && spec.kind != pnnx::EXPORTED_TENSOR_CONSTANT)
            continue;

        const std::map<std::string, pnnx::ExportedTensorMeta>::const_iterator meta_it = program.graph.tensor_values.find(spec.arg.name);
        if (meta_it == program.graph.tensor_values.end() || meta_it->second.dtype != 7)
        {
            fprintf(stderr, "real linear fixture has unexpected state metadata\n");
            return 1;
        }

        pnnx::MaterializedExportedTensor tensor;
        tensor.pnnx_type = 1;
        for (size_t j = 0; j < meta_it->second.sizes.size(); j++)
        {
            if (meta_it->second.sizes[j] < 0 || meta_it->second.sizes[j] > INT_MAX)
            {
                fprintf(stderr, "real linear fixture shape is out of range\n");
                return 1;
            }
            tensor.shape.push_back((int)meta_it->second.sizes[j]);
        }
        state[spec.target] = tensor;
    }

    pnnx::Graph graph;
    if (pnnx::lower_exported_program(program, state, graph, error) != 0)
    {
        fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }

    for (size_t i = 0; i < graph.ops.size(); i++)
    {
        const pnnx::Operator* op = graph.ops[i];
        fprintf(stdout, "before|%s|%s|%s|%s\n", op->type.c_str(), op->name.c_str(), join_operands(op->inputs).c_str(), join_operands(op->outputs).c_str());
    }

    pnnx::pass_level2(graph);
    fprintf(stdout, "after-types|");
    for (size_t i = 0; i < graph.ops.size(); i++)
    {
        if (i != 0)
            fprintf(stdout, ",");
        fprintf(stdout, "%s", graph.ops[i]->type.c_str());
    }
    fprintf(stdout, "\n");
    return 0;
}

int main(int argc, char** argv)
{
    if (argc == 3 && std::string(argv[1]) == "inspect")
        return inspect_package(argv[2]);
    if (argc != 1)
        return 2;

    test_linear_relu_graph();
    test_linear_default_bias_constant();
    test_conv2d_constant_arguments();
    test_batch_norm_constant_arguments();
    test_inplace_relu_is_functionalized_by_level2();
    test_max_pool2d_constant_arguments();
    test_inplace_add_is_functionalized_by_level2();
    test_adaptive_avg_pool2d_output_size();
    test_flatten_dimensions();
    test_dispatcher_backed_target_lowering();
    test_tensor_list_argument_lowering();
    test_string_and_memory_format_argument_lowering();
    test_reject_unsupported_signature_inputs();
    test_reject_non_inference_outputs();
    test_reject_invalid_graph_definitions();
    test_reject_signature_graph_mismatch();
    test_reject_state_metadata_mismatch();
    test_generated_names_do_not_collide();

    if (test_failures != 0)
    {
        fprintf(stderr, "%d exported program ir tests failed across %d paths\n", test_failures, test_paths);
        return 1;
    }

    fprintf(stdout, "exported program ir tests passed %d paths\n", test_paths);
    return 0;
}
