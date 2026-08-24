// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "load_exported_program.h"
#include "pass_level2.h"
#include "pass_level3/fuse_op1ton_unpack.h"
#include "pt2_archive.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

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

static pnnx::ExportedArgument make_complex(double real, double imag)
{
    pnnx::ExportedArgument argument;
    argument.type = pnnx::EXPORTED_ARGUMENT_COMPLEX;
    argument.complex_real_value = real;
    argument.complex_imag_value = imag;
    return argument;
}

static pnnx::ExportedArgument make_floats(const std::vector<double>& values)
{
    pnnx::ExportedArgument argument;
    argument.type = pnnx::EXPORTED_ARGUMENT_FLOAT_LIST;
    argument.float_values = values;
    return argument;
}

static pnnx::ExportedArgument make_int(int64_t value)
{
    pnnx::ExportedArgument argument;
    argument.type = pnnx::EXPORTED_ARGUMENT_INT;
    argument.int_value = value;
    return argument;
}

static pnnx::ExportedArgument make_symbolic(pnnx::ExportedArgumentType type, const std::string& name)
{
    pnnx::ExportedArgument argument;
    argument.type = type;
    argument.name = name;
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

static pnnx::ExportedArgument make_device(const std::string& type, int64_t index = 0, bool has_index = false)
{
    pnnx::ExportedArgument argument;
    argument.type = pnnx::EXPORTED_ARGUMENT_DEVICE;
    argument.device_value.type = type;
    argument.device_value.index = index;
    argument.device_value.has_index = has_index;
    return argument;
}

static pnnx::ExportedArgument make_graph(const std::string& name, const pnnx::ExportedGraph& graph)
{
    pnnx::ExportedArgument argument;
    argument.type = pnnx::EXPORTED_ARGUMENT_GRAPH;
    argument.graph_name = name;
    argument.graph_value.reset(new pnnx::ExportedGraph(graph));
    return argument;
}

static pnnx::ExportedTreeSpec make_output_leaf()
{
    pnnx::ExportedTreeSpec spec;
    spec.type = pnnx::EXPORTED_TREE_SPEC_LEAF;
    return spec;
}

static pnnx::ExportedTreeSpec make_output_tree(pnnx::ExportedTreeSpecType type, const std::vector<pnnx::ExportedTreeSpec>& children)
{
    pnnx::ExportedTreeSpec spec;
    spec.type = type;
    spec.children = children;
    return spec;
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
    for (size_t i = 0; i < sizes.size(); i++)
        meta.sizes.push_back(pnnx::ExportedSymInt(sizes[i]));
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

static pnnx::MaterializedExportedTensor make_float_state_tensor(const std::vector<int>& shape, const std::vector<float>& values)
{
    pnnx::MaterializedExportedTensor tensor = make_state_tensor(shape, values.size() * sizeof(float));
    memcpy(tensor.data.data(), values.data(), tensor.data.size());
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

static pnnx::ExportedProgram make_bounded_dynamic_result_program(bool legacy_range_node = false)
{
    pnnx::ExportedProgram program;
    program.header.schema_major = 8;
    program.header.schema_minor = legacy_range_node ? 14 : 20;
    program.header.torch_version = legacy_range_node ? "2.9.1" : "2.12.1+cu126";
    program.header.opset_version["aten"] = 10;

    program.graph.inputs.push_back(make_tensor("x"));

    pnnx::ExportedNode gt;
    gt.name = "gt";
    gt.has_name = true;
    gt.target = "torch.ops.aten.gt.Scalar";
    gt.inputs.push_back(make_input("self", "x"));
    gt.inputs.push_back(make_input("other", make_float(0.5)));
    gt.outputs.push_back(make_tensor("mask"));
    program.graph.nodes.push_back(gt);

    pnnx::ExportedNode masked_select;
    masked_select.name = "masked_select";
    masked_select.has_name = true;
    masked_select.target = "torch.ops.aten.masked_select.default";
    masked_select.inputs.push_back(make_input("self", "x"));
    masked_select.inputs.push_back(make_input("mask", "mask"));
    masked_select.outputs.push_back(make_tensor("selected"));
    program.graph.nodes.push_back(masked_select);

    pnnx::ExportedNode sym_size;
    sym_size.name = "sym_size_int";
    sym_size.has_name = true;
    sym_size.target = "torch.ops.aten.sym_size.int";
    sym_size.inputs.push_back(make_input("self", "selected"));
    sym_size.inputs.push_back(make_input("dim", make_int(0)));
    sym_size.outputs.push_back(make_symbolic(pnnx::EXPORTED_ARGUMENT_SYMBOLIC_INT, "sym_size_int"));
    program.graph.nodes.push_back(sym_size);

    if (legacy_range_node)
    {
        pnnx::ExportedNode constrain_range;
        constrain_range.target = "torch.ops.aten.sym_constrain_range_for_size.default";
        constrain_range.inputs.push_back(make_input("size", make_symbolic(pnnx::EXPORTED_ARGUMENT_SYMBOLIC_INT, "sym_size_int")));
        program.graph.nodes.push_back(constrain_range);
    }

    pnnx::ExportedNode ge;
    ge.name = "ge";
    ge.has_name = true;
    ge.target = "_operator.ge";
    ge.inputs.push_back(make_input("a", make_symbolic(pnnx::EXPORTED_ARGUMENT_SYMBOLIC_INT, "sym_size_int")));
    ge.inputs.push_back(make_input("b", make_int(0)));
    ge.outputs.push_back(make_symbolic(pnnx::EXPORTED_ARGUMENT_SYMBOLIC_BOOL, "ge"));
    program.graph.nodes.push_back(ge);

    pnnx::ExportedNode assert_min;
    assert_min.name = "assert_min";
    assert_min.has_name = true;
    assert_min.target = "torch.ops.aten._assert_scalar.default";
    assert_min.inputs.push_back(make_input("self", make_symbolic(pnnx::EXPORTED_ARGUMENT_SYMBOLIC_BOOL, "ge")));
    assert_min.inputs.push_back(make_input("assert_msg", make_string("Runtime assertion failed for expression u0 >= 0")));
    program.graph.nodes.push_back(assert_min);

    pnnx::ExportedNode le = ge;
    le.name = "le";
    le.target = "_operator.le";
    le.inputs[1] = make_input("b", make_int(48));
    le.outputs[0] = make_symbolic(pnnx::EXPORTED_ARGUMENT_SYMBOLIC_BOOL, "le");
    program.graph.nodes.push_back(le);

    pnnx::ExportedNode assert_max = assert_min;
    assert_max.name = "assert_max";
    assert_max.inputs[0] = make_input("self", make_symbolic(pnnx::EXPORTED_ARGUMENT_SYMBOLIC_BOOL, "le"));
    assert_max.inputs[1] = make_input("assert_msg", make_string("Runtime assertion failed for expression u0 <= 48"));
    program.graph.nodes.push_back(assert_max);

    program.graph.outputs.push_back(make_tensor("selected"));
    program.graph.tensor_values["x"] = make_tensor_meta(std::vector<int64_t>{1, 3, 16});
    program.graph.tensor_values["mask"] = make_tensor_meta(std::vector<int64_t>{1, 3, 16});
    program.graph.tensor_values["mask"].dtype = 12;
    program.graph.tensor_values["selected"] = make_tensor_meta(std::vector<int64_t>{1});
    program.graph.tensor_values["selected"].sizes[0].type = pnnx::EXPORTED_SYM_INT_EXPRESSION;
    program.graph.tensor_values["selected"].sizes[0].expression = "Symbol('u0', integer=True)";

    pnnx::ExportedSymInt sym_int;
    sym_int.type = pnnx::EXPORTED_SYM_INT_EXPRESSION;
    sym_int.expression = "Symbol('u0', integer=True)";
    program.graph.sym_int_values["sym_size_int"] = sym_int;

    pnnx::ExportedSymBool sym_bool;
    sym_bool.is_expression = true;
    sym_bool.expression = "GreaterThan(Symbol('u0', integer=True), Integer(0))";
    program.graph.sym_bool_values["ge"] = sym_bool;
    sym_bool.expression = "LessThan(Symbol('u0', integer=True), Integer(48))";
    program.graph.sym_bool_values["le"] = sym_bool;

    pnnx::ExportedRangeConstraint constraint;
    constraint.has_min = true;
    constraint.min = 0;
    constraint.has_max = true;
    constraint.max = 48;
    program.range_constraints["u0"] = constraint;

    pnnx::ExportedInputSpec input;
    input.kind = pnnx::EXPORTED_USER_INPUT;
    input.arg = make_tensor("x");
    program.input_specs.push_back(input);

    pnnx::ExportedOutputSpec output;
    output.kind = pnnx::EXPORTED_USER_OUTPUT;
    output.arg = make_tensor("selected");
    program.output_specs.push_back(output);

    return program;
}

static pnnx::ExportedProgram make_higher_order_program(bool autocast_enabled = false)
{
    pnnx::ExportedProgram program;
    program.header.schema_major = 8;
    program.header.schema_minor = 20;
    program.header.torch_version = "2.12.1+cu126";
    program.header.opset_version["aten"] = 10;

    program.graph.inputs.push_back(make_tensor("x"));

    pnnx::ExportedNode neg;
    neg.name = "neg";
    neg.has_name = true;
    neg.target = "torch.ops.aten.neg.default";
    neg.inputs.push_back(make_input("self", "x"));
    neg.outputs.push_back(make_tensor("inner_input"));
    program.graph.nodes.push_back(neg);

    pnnx::ExportedGraph autocast_graph;
    autocast_graph.inputs.push_back(make_tensor("sub_input"));
    pnnx::ExportedNode sigmoid;
    sigmoid.name = "sigmoid";
    sigmoid.has_name = true;
    sigmoid.target = "torch.ops.aten.sigmoid.default";
    sigmoid.inputs.push_back(make_input("self", "sub_input"));
    sigmoid.outputs.push_back(make_tensor("sub_output"));
    autocast_graph.nodes.push_back(sigmoid);
    autocast_graph.outputs.push_back(make_tensor("sub_output"));
    autocast_graph.tensor_values["sub_input"] = make_tensor_meta(std::vector<int64_t>{2, 4});
    autocast_graph.tensor_values["sub_output"] = make_tensor_meta(std::vector<int64_t>{2, 4});

    pnnx::ExportedGraph grad_graph;
    grad_graph.inputs.push_back(make_tensor("captured"));
    pnnx::ExportedNode relu;
    relu.name = "relu";
    relu.has_name = true;
    relu.target = "torch.ops.aten.relu.default";
    relu.inputs.push_back(make_input("self", "captured"));
    relu.outputs.push_back(make_tensor("inner_input"));
    grad_graph.nodes.push_back(relu);

    pnnx::ExportedNode metadata_assert;
    metadata_assert.name = "metadata_assert";
    metadata_assert.has_name = true;
    metadata_assert.target = "torch.ops.aten._assert_tensor_metadata.default";
    metadata_assert.inputs.push_back(make_input("a", "inner_input"));
    grad_graph.nodes.push_back(metadata_assert);

    pnnx::ExportedNode autocast;
    autocast.name = "autocast";
    autocast.has_name = true;
    autocast.target = "torch.ops.higher_order.wrap_with_autocast";
    autocast.inputs.push_back(make_input("", make_string("cpu")));
    autocast.inputs.push_back(make_input("", make_enum(pnnx::EXPORTED_ARGUMENT_SCALAR_TYPE, 13)));
    autocast.inputs.push_back(make_input("", make_bool(autocast_enabled)));
    autocast.inputs.push_back(make_input("", make_bool(false)));
    autocast.inputs.push_back(make_input("", make_graph("autocast_subgraph", autocast_graph)));
    autocast.inputs.push_back(make_input("", make_tensor("inner_input")));
    autocast.outputs.push_back(make_tensor("inner_output"));
    grad_graph.nodes.push_back(autocast);
    grad_graph.outputs.push_back(make_tensor("inner_output"));
    grad_graph.tensor_values["captured"] = make_tensor_meta(std::vector<int64_t>{2, 4});
    grad_graph.tensor_values["inner_input"] = make_tensor_meta(std::vector<int64_t>{2, 4});
    grad_graph.tensor_values["inner_output"] = make_tensor_meta(std::vector<int64_t>{2, 4});

    pnnx::ExportedNode set_grad;
    set_grad.name = "set_grad";
    set_grad.has_name = true;
    set_grad.target = "torch.ops.higher_order.wrap_with_set_grad_enabled";
    set_grad.inputs.push_back(make_input("", make_bool(false)));
    set_grad.inputs.push_back(make_input("", make_graph("grad_subgraph", grad_graph)));
    set_grad.inputs.push_back(make_input("", make_tensor("inner_input")));
    set_grad.outputs.push_back(make_tensor("z"));
    program.graph.nodes.push_back(set_grad);

    program.graph.outputs.push_back(make_tensor("z"));
    program.graph.tensor_values["x"] = make_tensor_meta(std::vector<int64_t>{2, 4});
    program.graph.tensor_values["inner_input"] = make_tensor_meta(std::vector<int64_t>{2, 4});
    program.graph.tensor_values["z"] = make_tensor_meta(std::vector<int64_t>{2, 4});

    pnnx::ExportedInputSpec input;
    input.kind = pnnx::EXPORTED_USER_INPUT;
    input.arg = make_tensor("x");
    program.input_specs.push_back(input);

    pnnx::ExportedOutputSpec output;
    output.kind = pnnx::EXPORTED_USER_OUTPUT;
    output.arg = make_tensor("z");
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

static pnnx::ExportedProgram make_chunk_program()
{
    pnnx::ExportedProgram program;
    program.header.schema_major = 8;
    program.header.schema_minor = 20;
    program.header.torch_version = "2.12.1+cu126";
    program.header.opset_version["aten"] = 10;
    program.graph.inputs.push_back(make_tensor("x"));

    pnnx::ExportedNode chunk;
    chunk.name = "chunk";
    chunk.has_name = true;
    chunk.target = "torch.ops.aten.chunk.default";
    chunk.inputs.push_back(make_input("self", make_tensor("x")));
    chunk.inputs.push_back(make_input("chunks", make_int(2)));
    chunk.inputs.push_back(make_input("dim", make_int(1)));
    chunk.outputs.push_back(make_tensors(std::vector<std::string>{"chunk_0", "chunk_1"}));
    program.graph.nodes.push_back(chunk);
    program.graph.outputs.push_back(make_tensor("chunk_0"));
    program.graph.outputs.push_back(make_tensor("chunk_1"));

    program.graph.tensor_values["x"] = make_tensor_meta(std::vector<int64_t>{1, 4});
    program.graph.tensor_values["chunk_0"] = make_tensor_meta(std::vector<int64_t>{1, 2});
    program.graph.tensor_values["chunk_1"] = make_tensor_meta(std::vector<int64_t>{1, 2});

    pnnx::ExportedInputSpec input;
    input.kind = pnnx::EXPORTED_USER_INPUT;
    input.arg = make_tensor("x");
    program.input_specs.push_back(input);

    const char* output_names[] = {"chunk_0", "chunk_1"};
    for (size_t i = 0; i < sizeof(output_names) / sizeof(output_names[0]); i++)
    {
        pnnx::ExportedOutputSpec output;
        output.kind = pnnx::EXPORTED_USER_OUTPUT;
        output.arg = make_tensor(output_names[i]);
        program.output_specs.push_back(output);
    }
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

static pnnx::ExportedProgram make_weight_norm_program(bool static_weight)
{
    pnnx::ExportedProgram program;
    program.header.schema_major = 8;
    program.header.schema_minor = 20;
    program.header.torch_version = "2.12.1+cu126";
    program.header.opset_version["aten"] = 10;

    program.graph.inputs.push_back(make_tensor("weight_v"));
    program.graph.inputs.push_back(make_tensor("weight_g"));

    pnnx::ExportedNode weight_norm;
    weight_norm.name = "_weight_norm";
    weight_norm.has_name = true;
    weight_norm.target = "torch.ops.aten._weight_norm.default";
    weight_norm.inputs.push_back(make_input("v", make_tensor("weight_v")));
    weight_norm.inputs.push_back(make_input("g", make_tensor("weight_g")));
    weight_norm.outputs.push_back(make_tensor("normalized_weight"));
    program.graph.nodes.push_back(weight_norm);
    program.graph.outputs.push_back(make_tensor("normalized_weight"));

    program.graph.tensor_values["weight_v"] = make_tensor_meta(std::vector<int64_t>{2, 3});
    program.graph.tensor_values["weight_g"] = make_tensor_meta(std::vector<int64_t>{2, 1});
    program.graph.tensor_values["normalized_weight"] = make_tensor_meta(std::vector<int64_t>{2, 3});

    const std::vector<std::string> input_names = {"weight_v", "weight_g"};
    for (size_t i = 0; i < input_names.size(); i++)
    {
        pnnx::ExportedInputSpec input;
        input.kind = static_weight ? pnnx::EXPORTED_PARAMETER : pnnx::EXPORTED_USER_INPUT;
        input.arg = make_tensor(input_names[i]);
        if (static_weight)
            input.target = input_names[i];
        program.input_specs.push_back(input);
    }

    pnnx::ExportedOutputSpec output;
    output.kind = pnnx::EXPORTED_USER_OUTPUT;
    output.arg = make_tensor("normalized_weight");
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

static void expect_lower_error(const pnnx::ExportedProgram& program,
                               const std::map<std::string, pnnx::MaterializedExportedTensor>& state,
                               const std::string& expected_error,
                               const char* name);

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

static void test_bounded_dynamic_result_shape()
{
    const pnnx::ExportedProgram program = make_bounded_dynamic_result_program();
    pnnx::Graph graph;
    std::string error = "stale";
    const int result = pnnx::lower_exported_program(program, std::map<std::string, pnnx::MaterializedExportedTensor>(), graph, error);

    check(result == 0, "bounded dynamic result", "lowering failed: " + error);
    check(error.empty(), "bounded dynamic result", "success retained an error");
    if (result == 0)
    {
        check(count_operator(graph, "aten::sym_size") == 0 && count_operator(graph, "aten::_assert_scalar") == 0, "bounded dynamic result", "shape constraint nodes were retained");
        pnnx::Operand* selected = graph.get_operand("selected");
        check(selected && selected->shape == std::vector<int>({48}), "bounded dynamic result", "finite upper bound was not used as output shape");
        check(selected && selected->producer && selected->producer->type == "aten::masked_select", "bounded dynamic result", "masked_select producer changed");
    }

    pnnx::ExportedProgram missing_bound = make_bounded_dynamic_result_program();
    missing_bound.range_constraints.clear();
    expect_lower_error(missing_bound, std::map<std::string, pnnx::MaterializedExportedTensor>(), "symbolic size u0 has no finite upper bound", "dynamic result missing bound");

    pnnx::ExportedProgram complex_expression = make_bounded_dynamic_result_program();
    complex_expression.graph.tensor_values["selected"].sizes[0].expression = "Add(Symbol('u0', integer=True), Integer(1))";
    expect_lower_error(complex_expression, std::map<std::string, pnnx::MaterializedExportedTensor>(), "unsupported symbolic size Add", "dynamic result complex expression");

    pnnx::ExportedProgram oversized_bound = make_bounded_dynamic_result_program();
    oversized_bound.range_constraints["u0"].max = (int64_t)INT_MAX + 1;
    expect_lower_error(oversized_bound, std::map<std::string, pnnx::MaterializedExportedTensor>(), "upper bound does not fit pnnx", "dynamic result oversized bound");

    const pnnx::ExportedProgram legacy_program = make_bounded_dynamic_result_program(true);
    pnnx::Graph legacy_graph;
    error = "stale";
    const int legacy_result = pnnx::lower_exported_program(legacy_program, std::map<std::string, pnnx::MaterializedExportedTensor>(), legacy_graph, error);
    check(legacy_result == 0, "bounded dynamic result schema 8.14", "lowering failed: " + error);
    check(error.empty(), "bounded dynamic result schema 8.14", "success retained an error");
    check(legacy_result == 0 && count_operator(legacy_graph, "aten::sym_constrain_range_for_size") == 0, "bounded dynamic result schema 8.14", "legacy range constraint node was retained");
}

static void test_higher_order_graph_lowering()
{
    const pnnx::ExportedProgram program = make_higher_order_program();
    pnnx::Graph graph;
    std::string error = "stale";
    const int result = pnnx::lower_exported_program(program, std::map<std::string, pnnx::MaterializedExportedTensor>(), graph, error);

    check(result == 0, "higher order graph", "lowering failed: " + error);
    check(error.empty(), "higher order graph", "success retained an error");
    if (result != 0)
        return;

    check(graph.ops.size() == 5, "higher order graph", "wrong operator count");
    check(count_operator(graph, "aten::neg") == 1, "higher order graph", "missing outer aten::neg");
    check(count_operator(graph, "aten::relu") == 1, "higher order graph", "missing inlined aten::relu");
    check(count_operator(graph, "aten::sigmoid") == 1, "higher order graph", "missing recursively inlined aten::sigmoid");
    check(count_operator(graph, "aten::_assert_tensor_metadata") == 0, "higher order graph", "metadata assertion was not eliminated");

    pnnx::Operator* neg = find_operator(graph, "aten::neg");
    pnnx::Operator* relu = find_operator(graph, "aten::relu");
    pnnx::Operator* sigmoid = find_operator(graph, "aten::sigmoid");
    check(neg && neg->outputs.size() == 1 && neg->outputs[0]->name == "inner_input", "higher order graph", "outer value name changed");
    check(relu && relu->inputs.size() == 1 && relu->inputs[0]->producer == neg, "higher order graph", "captured input was not bound to subgraph input");
    check(relu && relu->outputs.size() == 1 && relu->outputs[0]->name != "inner_input", "higher order graph", "colliding subgraph value was not renamed");
    check(sigmoid && sigmoid->inputs.size() == 1 && sigmoid->inputs[0]->producer == relu, "higher order graph", "nested subgraph input was not rebound");
    check(sigmoid && sigmoid->outputs.size() == 1 && sigmoid->outputs[0]->name == "z", "higher order graph", "wrapper output mapping changed");
    check(sigmoid && sigmoid->outputs.size() == 1 && sigmoid->outputs[0]->consumers.size() == 1 && sigmoid->outputs[0]->consumers[0]->type == "pnnx.Output", "higher order graph", "mapped wrapper output is not returned");

    pnnx::ExportedProgram enabled_autocast = make_higher_order_program(true);
    expect_lower_error(enabled_autocast, std::map<std::string, pnnx::MaterializedExportedTensor>(), "enabled autocast higher-order graph is unsupported", "enabled autocast graph");

    pnnx::ExportedProgram enabled_set_grad = make_higher_order_program();
    enabled_set_grad.graph.nodes[1].inputs[0].arg.bool_value = true;
    expect_lower_error(enabled_set_grad, std::map<std::string, pnnx::MaterializedExportedTensor>(), "enabled set-grad higher-order graph is unsupported", "enabled set grad graph");

    pnnx::ExportedProgram keyword_wrapper = make_higher_order_program();
    keyword_wrapper.graph.nodes[1].inputs[0].kind = pnnx::EXPORTED_ARGUMENT_KIND_KEYWORD;
    expect_lower_error(keyword_wrapper, std::map<std::string, pnnx::MaterializedExportedTensor>(), "higher-order wrapper arguments must be positional", "keyword higher order wrapper");

    pnnx::ExportedProgram missing_capture = make_higher_order_program();
    missing_capture.graph.nodes[1].inputs.pop_back();
    expect_lower_error(missing_capture, std::map<std::string, pnnx::MaterializedExportedTensor>(), "captured argument count does not match subgraph input count", "missing higher order capture");

    pnnx::ExportedProgram mismatched_metadata = make_higher_order_program();
    mismatched_metadata.graph.nodes[1].inputs[1].arg.graph_value->tensor_values["captured"] = make_tensor_meta(std::vector<int64_t>{1, 4});
    expect_lower_error(mismatched_metadata, std::map<std::string, pnnx::MaterializedExportedTensor>(), "subgraph tensor metadata does not match bound value", "higher order metadata mismatch");
}

static void test_complex_scalar_lowering()
{
    pnnx::ExportedProgram program;
    program.header.schema_major = 8;
    program.header.schema_minor = 20;
    program.header.torch_version = "2.12.1+cu126";
    program.header.opset_version["aten"] = 10;

    program.graph.inputs.push_back(make_tensor("x"));

    pnnx::ExportedNode sub;
    sub.name = "sub";
    sub.has_name = true;
    sub.target = "torch.ops.aten.sub.Tensor";
    sub.inputs.push_back(make_input("self", "x"));
    sub.inputs.push_back(make_input("other", make_complex(0.0, 4.0)));
    sub.outputs.push_back(make_tensor("sub"));
    program.graph.nodes.push_back(sub);
    program.graph.outputs.push_back(make_tensor("sub"));

    pnnx::ExportedTensorMeta meta = make_tensor_meta(std::vector<int64_t>{3, 15});
    meta.dtype = 10;
    program.graph.tensor_values["x"] = meta;
    program.graph.tensor_values["sub"] = meta;

    pnnx::ExportedInputSpec input;
    input.kind = pnnx::EXPORTED_USER_INPUT;
    input.arg = make_tensor("x");
    program.input_specs.push_back(input);

    pnnx::ExportedOutputSpec output;
    output.kind = pnnx::EXPORTED_USER_OUTPUT;
    output.arg = make_tensor("sub");
    program.output_specs.push_back(output);

    pnnx::Graph graph;
    std::string error = "stale";
    const int result = pnnx::lower_exported_program(program, std::map<std::string, pnnx::MaterializedExportedTensor>(), graph, error);

    check(result == 0, "complex scalar", "lowering failed: " + error);
    check(error.empty(), "complex scalar", "success retained an error");
    if (result != 0)
        return;

    pnnx::Operator* op = find_operator(graph, "aten::sub");
    check(op && op->inputs.size() == 3, "complex scalar", "sub inputs are not canonical");
    check(op && op->inputs.size() == 3 && op->inputs[1]->producer && op->inputs[1]->producer->type == "prim::Constant", "complex scalar", "complex argument is not a constant");
    if (!op || op->inputs.size() != 3 || !op->inputs[1]->producer || !op->inputs[1]->producer->has_param("value"))
        return;

    const pnnx::Parameter& complex_value = op->inputs[1]->producer->params.at("value");
    const pnnx::Parameter& alpha = op->inputs[2]->producer->params.at("value");
    check(complex_value.type == 10 && complex_value.c.real() == 0.f && complex_value.c.imag() == 4.f, "complex scalar", "complex constant value changed");
    check(alpha.type == 2 && alpha.i == 1, "complex scalar", "default alpha changed");
    check(op->inputs[0]->type == 10 && op->outputs.size() == 1 && op->outputs[0]->type == 10, "complex scalar", "complex tensor metadata changed");
}

static void test_output_tree_lowering()
{
    {
        pnnx::ExportedProgram program = make_linear_relu_program();
        program.output_tree_spec = make_output_leaf();

        pnnx::Graph graph;
        std::string error;
        const int result = pnnx::lower_exported_program(program, make_linear_state(), graph, error);

        check(result == 0, "output tree leaf", error);
        check(result == 0 && count_operator(graph, "prim::TupleConstruct") == 0 && count_operator(graph, "pnnx.Output") == 1, "output tree leaf", "leaf output graph changed");
    }

    {
        pnnx::ExportedProgram program = make_linear_relu_program();
        program.output_tree_spec = make_output_tree(pnnx::EXPORTED_TREE_SPEC_TUPLE, std::vector<pnnx::ExportedTreeSpec>(1, make_output_leaf()));

        pnnx::Graph graph;
        std::string error;
        const int result = pnnx::lower_exported_program(program, make_linear_state(), graph, error);

        check(result == 0, "output tree one tuple", error);
        pnnx::Operator* tuple = find_operator(graph, "prim::TupleConstruct");
        pnnx::Operator* output = find_operator(graph, "pnnx.Output");
        check(tuple && tuple->inputs.size() == 1 && tuple->inputs[0]->name == "relu" && tuple->outputs.size() == 1, "output tree one tuple", "tuple construct wiring changed");
        check(tuple && output && output->inputs.size() == 1 && tuple->outputs[0] == output->inputs[0], "output tree one tuple", "tuple output wiring changed");
        check(count_operator(graph, "pnnx.Output") == 1, "output tree one tuple", "tuple produced multiple graph outputs");
    }

    {
        pnnx::ExportedProgram program = make_linear_relu_program();
        const pnnx::ExportedTreeSpec list = make_output_tree(pnnx::EXPORTED_TREE_SPEC_LIST, std::vector<pnnx::ExportedTreeSpec>(1, make_output_leaf()));
        program.output_tree_spec = make_output_tree(pnnx::EXPORTED_TREE_SPEC_TUPLE, std::vector<pnnx::ExportedTreeSpec>(1, list));

        pnnx::Graph graph;
        std::string error;
        const int result = pnnx::lower_exported_program(program, make_linear_state(), graph, error);

        check(result == 0, "output tree nested", error);
        pnnx::Operator* list_construct = find_operator(graph, "prim::ListConstruct");
        pnnx::Operator* tuple_construct = find_operator(graph, "prim::TupleConstruct");
        pnnx::Operator* output = find_operator(graph, "pnnx.Output");
        check(list_construct && tuple_construct && list_construct->outputs.size() == 1 && tuple_construct->inputs.size() == 1 && tuple_construct->inputs[0] == list_construct->outputs[0], "output tree nested", "nested container wiring changed");
        check(tuple_construct && output && output->inputs.size() == 1 && output->inputs[0] == tuple_construct->outputs[0], "output tree nested", "nested output wiring changed");
    }

    {
        pnnx::ExportedProgram program = make_linear_relu_program();
        program.output_tree_spec = make_output_tree(pnnx::EXPORTED_TREE_SPEC_TUPLE, std::vector<pnnx::ExportedTreeSpec>(2, make_output_leaf()));
        expect_lower_error(program, make_linear_state(), "output treespec leaf count does not match graph outputs", "output tree leaf mismatch");
    }
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

static void test_conv_nd_direct_lowering()
{
    struct ConvCase
    {
        const char* target;
        const char* aten_type;
        const char* canonical_type;
        std::vector<int64_t> weight_shape;
        std::vector<int64_t> input_shape;
        std::vector<int64_t> output_shape;
        std::vector<int> default_values;
        bool string_padding;
    };

    const ConvCase cases[] = {
        {"torch.ops.aten.conv1d.default", "aten::conv1d", "F.conv1d", std::vector<int64_t>{3, 4, 3}, std::vector<int64_t>{1, 4, 8}, std::vector<int64_t>{1, 3, 6}, std::vector<int>{1}, false},
        {"torch.ops.aten.conv1d.padding", "aten::conv1d", "F.conv1d", std::vector<int64_t>{3, 4, 3}, std::vector<int64_t>{1, 4, 8}, std::vector<int64_t>{1, 3, 8}, std::vector<int>{1}, true},
        {"torch.ops.aten.conv3d.default", "aten::conv3d", "F.conv3d", std::vector<int64_t>{3, 4, 3, 3, 3}, std::vector<int64_t>{1, 4, 8, 8, 8}, std::vector<int64_t>{1, 3, 6, 6, 6}, std::vector<int>{1, 1, 1}, false},
        {"torch.ops.aten.conv3d.padding", "aten::conv3d", "F.conv3d", std::vector<int64_t>{3, 4, 3, 3, 3}, std::vector<int64_t>{1, 4, 8, 8, 8}, std::vector<int64_t>{1, 3, 8, 8, 8}, std::vector<int>{1, 1, 1}, true},
    };

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
    {
        const ConvCase& c = cases[i];
        pnnx::ExportedProgram program = make_conv2d_program();
        program.graph.nodes[0].target = c.target;
        if (c.string_padding)
            program.graph.nodes[0].inputs.push_back(make_keyword_input("padding", make_string("same")));
        program.graph.tensor_values["p_conv_weight"] = make_tensor_meta(c.weight_shape);
        program.graph.tensor_values["x"] = make_tensor_meta(c.input_shape);
        program.graph.tensor_values["conv2d"] = make_tensor_meta(c.output_shape);

        std::vector<int> weight_shape(c.weight_shape.begin(), c.weight_shape.end());
        size_t weight_count = 1;
        for (size_t j = 0; j < c.weight_shape.size(); j++)
            weight_count *= (size_t)c.weight_shape[j];
        std::map<std::string, pnnx::MaterializedExportedTensor> state;
        state["conv.weight"] = make_state_tensor(weight_shape, weight_count * 4);

        pnnx::Graph graph;
        std::string error = "stale";
        const int result = pnnx::lower_exported_program(program, state, graph, error);

        check(result == 0, "conv nd direct", "lowering failed: " + error);
        check(error.empty(), "conv nd direct", "success retained an error");
        if (result != 0)
            continue;

        pnnx::Operator* conv = find_operator(graph, c.aten_type);
        check(conv && conv->inputnames == std::vector<std::string>({"input", "weight", "bias", "stride", "padding", "dilation", "groups"}), "conv nd direct", "canonical argument names changed");
        check(conv && conv->inputs.size() == 7, "conv nd direct", "conv does not have seven canonical inputs");
        if (!conv || conv->inputs.size() != 7)
            continue;

        const pnnx::Parameter& stride = conv->inputs[3]->producer->params.at("value");
        const pnnx::Parameter& padding = conv->inputs[4]->producer->params.at("value");
        const pnnx::Parameter& dilation = conv->inputs[5]->producer->params.at("value");
        check(stride.type == 5 && stride.ai == c.default_values, "conv nd direct", "stride default changed");
        check(dilation.type == 5 && dilation.ai == c.default_values, "conv nd direct", "dilation default changed");
        check(c.string_padding ? padding.type == 4 && padding.s == "same" : padding.type == 5 && padding.ai == std::vector<int>(c.default_values.size(), 0), "conv nd direct", "padding changed");

        pnnx::pass_level2(graph);
        pnnx::Operator* canonical = find_operator(graph, c.canonical_type);
        check(find_operator(graph, c.aten_type) == 0 && canonical != 0, "conv nd direct pass level2", "direct convolution was not canonicalized");
        check(canonical && canonical->inputnames == std::vector<std::string>({"input", "weight", "bias", "stride", "padding", "dilation", "groups"}) && canonical->inputs.size() == 7, "conv nd direct pass level2", "canonical inputs changed");
        if (canonical && canonical->inputs.size() == 7)
        {
            const pnnx::Parameter& canonical_padding = canonical->inputs[4]->producer->params.at("value");
            check(c.string_padding ? canonical_padding.type == 4 && canonical_padding.s == "same" : canonical_padding.type == 5 && canonical_padding.ai == std::vector<int>(c.default_values.size(), 0), "conv nd direct pass level2", "canonical padding changed");
        }
    }
}

static void test_conv_transpose_nd_direct_lowering()
{
    struct ConvTransposeCase
    {
        const char* target;
        const char* aten_type;
        const char* canonical_type;
        std::vector<int64_t> weight_shape;
        std::vector<int64_t> input_shape;
        std::vector<int64_t> output_shape;
        std::vector<int> unit_values;
    };

    const ConvTransposeCase cases[] = {
        {"torch.ops.aten.conv_transpose1d.default", "aten::conv_transpose1d", "F.conv_transpose1d", std::vector<int64_t>{4, 3, 3}, std::vector<int64_t>{1, 4, 8}, std::vector<int64_t>{1, 3, 10}, std::vector<int>{1}},
        {"torch.ops.aten.conv_transpose2d.input", "aten::conv_transpose2d", "F.conv_transpose2d", std::vector<int64_t>{4, 3, 3, 3}, std::vector<int64_t>{1, 4, 8, 8}, std::vector<int64_t>{1, 3, 10, 10}, std::vector<int>{1, 1}},
        {"torch.ops.aten.conv_transpose3d.input", "aten::conv_transpose3d", "F.conv_transpose3d", std::vector<int64_t>{4, 3, 3, 3, 3}, std::vector<int64_t>{1, 4, 8, 8, 8}, std::vector<int64_t>{1, 3, 10, 10, 10}, std::vector<int>{1, 1, 1}},
    };

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
    {
        const ConvTransposeCase& c = cases[i];
        pnnx::ExportedProgram program = make_conv2d_program();
        program.graph.nodes[0].target = c.target;
        program.graph.tensor_values["p_conv_weight"] = make_tensor_meta(c.weight_shape);
        program.graph.tensor_values["x"] = make_tensor_meta(c.input_shape);
        program.graph.tensor_values["conv2d"] = make_tensor_meta(c.output_shape);

        std::vector<int> weight_shape(c.weight_shape.begin(), c.weight_shape.end());
        size_t weight_count = 1;
        for (size_t j = 0; j < c.weight_shape.size(); j++)
            weight_count *= (size_t)c.weight_shape[j];
        std::map<std::string, pnnx::MaterializedExportedTensor> state;
        state["conv.weight"] = make_state_tensor(weight_shape, weight_count * 4);

        pnnx::Graph graph;
        std::string error = "stale";
        const int result = pnnx::lower_exported_program(program, state, graph, error);

        check(result == 0, "conv transpose nd direct", "lowering failed: " + error);
        check(error.empty(), "conv transpose nd direct", "success retained an error");
        if (result != 0)
            continue;

        pnnx::Operator* conv = find_operator(graph, c.aten_type);
        check(conv && conv->inputnames == std::vector<std::string>({"input", "weight", "bias", "stride", "padding", "output_padding", "groups", "dilation"}), "conv transpose nd direct", "canonical argument names changed");
        check(conv && conv->inputs.size() == 8, "conv transpose nd direct", "conv transpose does not have eight canonical inputs");
        if (!conv || conv->inputs.size() != 8)
            continue;

        const pnnx::Parameter& bias = conv->inputs[2]->producer->params.at("value");
        const pnnx::Parameter& stride = conv->inputs[3]->producer->params.at("value");
        const pnnx::Parameter& padding = conv->inputs[4]->producer->params.at("value");
        const pnnx::Parameter& output_padding = conv->inputs[5]->producer->params.at("value");
        const pnnx::Parameter& groups = conv->inputs[6]->producer->params.at("value");
        const pnnx::Parameter& dilation = conv->inputs[7]->producer->params.at("value");
        check(bias.type == 0, "conv transpose nd direct", "bias default changed");
        check(stride.type == 5 && stride.ai == c.unit_values, "conv transpose nd direct", "stride default changed");
        check(padding.type == 5 && padding.ai == std::vector<int>(c.unit_values.size(), 0), "conv transpose nd direct", "padding default changed");
        check(output_padding.type == 5 && output_padding.ai == std::vector<int>(c.unit_values.size(), 0), "conv transpose nd direct", "output padding default changed");
        check(groups.type == 2 && groups.i == 1, "conv transpose nd direct", "groups default changed");
        check(dilation.type == 5 && dilation.ai == c.unit_values, "conv transpose nd direct", "dilation default changed");

        pnnx::pass_level2(graph);
        pnnx::Operator* canonical = find_operator(graph, c.canonical_type);
        check(find_operator(graph, c.aten_type) == 0 && canonical != 0, "conv transpose nd direct pass level2", "direct transposed convolution was not canonicalized");
        check(canonical && canonical->inputnames == std::vector<std::string>({"input", "weight", "bias", "stride", "padding", "output_padding", "groups", "dilation"}) && canonical->inputs.size() == 8, "conv transpose nd direct pass level2", "canonical inputs changed");
        if (canonical && canonical->inputs.size() == 8)
        {
            const pnnx::Parameter& canonical_output_padding = canonical->inputs[5]->producer->params.at("value");
            const pnnx::Parameter& canonical_dilation = canonical->inputs[7]->producer->params.at("value");
            check(canonical_output_padding.type == 5 && canonical_output_padding.ai == std::vector<int>(c.unit_values.size(), 0), "conv transpose nd direct pass level2", "canonical output padding changed");
            check(canonical_dilation.type == 5 && canonical_dilation.ai == c.unit_values, "conv transpose nd direct pass level2", "canonical dilation changed");
        }
    }
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

static void test_instance_norm_running_stats()
{
    pnnx::ExportedProgram program = make_batch_norm_program();
    pnnx::ExportedNode& instance_norm_node = program.graph.nodes[0];
    instance_norm_node.target = "torch.ops.aten.instance_norm.default";
    instance_norm_node.inputs[5].name = "use_input_stats";

    std::map<std::string, pnnx::MaterializedExportedTensor> state;
    state["weight"] = make_state_tensor(std::vector<int>{3}, 12);
    state["bias"] = make_state_tensor(std::vector<int>{3}, 12);
    state["running_mean"] = make_state_tensor(std::vector<int>{3}, 12);
    state["running_var"] = make_state_tensor(std::vector<int>{3}, 12);

    pnnx::Graph graph;
    std::string error = "stale";
    const int result = pnnx::lower_exported_program(program, state, graph, error);

    check(result == 0, "instance norm running stats", "lowering failed: " + error);
    check(error.empty(), "instance norm running stats", "success retained an error");
    if (result != 0)
        return;

    pnnx::Operator* instance_norm = find_operator(graph, "aten::instance_norm");
    check(instance_norm != 0, "instance norm running stats", "missing aten::instance_norm");
    check(instance_norm && instance_norm->inputnames == std::vector<std::string>({"input", "weight", "bias", "running_mean", "running_var", "use_input_stats", "momentum", "eps", "cudnn_enabled"}), "instance norm running stats", "instance_norm input names are not canonical");
    check(instance_norm && instance_norm->inputs.size() == 9, "instance norm running stats", "instance_norm does not have nine inputs");
    if (!instance_norm || instance_norm->inputs.size() != 9)
        return;

    const pnnx::Parameter& use_input_stats = instance_norm->inputs[5]->producer->params.at("value");
    const pnnx::Parameter& momentum = instance_norm->inputs[6]->producer->params.at("value");
    const pnnx::Parameter& eps = instance_norm->inputs[7]->producer->params.at("value");
    const pnnx::Parameter& cudnn_enabled = instance_norm->inputs[8]->producer->params.at("value");
    check(use_input_stats.type == 1 && !use_input_stats.b, "instance norm running stats", "use_input_stats constant is wrong");
    check(momentum.type == 3 && momentum.f == 0.1f, "instance norm running stats", "momentum constant is wrong");
    check(eps.type == 3 && eps.f == 1e-5f, "instance norm running stats", "eps constant is wrong");
    check(cudnn_enabled.type == 1 && !cudnn_enabled.b, "instance norm running stats", "cudnn_enabled constant is wrong");

    pnnx::pass_level2(graph);
    pnnx::Operator* canonical = find_operator(graph, "F.instance_norm");
    check(find_operator(graph, "aten::instance_norm") == 0 && canonical != 0, "instance norm running stats pass level2", "instance_norm with running stats was not canonicalized");
    check(canonical && canonical->inputs.size() == 5 && canonical->inputnames == std::vector<std::string>({"input", "running_mean", "running_var", "weight", "bias"}), "instance norm running stats pass level2", "canonical tensor inputs changed");
    check(canonical && canonical->has_param("use_input_stats") && canonical->params.at("use_input_stats").type == 1 && !canonical->params.at("use_input_stats").b, "instance norm running stats pass level2", "canonical use_input_stats is not false");
    check(canonical && canonical->has_param("eps") && canonical->params.at("eps").type == 3 && canonical->params.at("eps").f == 1e-5f, "instance norm running stats pass level2", "canonical eps changed");
}

static void test_weight_norm_lowering()
{
    const pnnx::ExportedProgram program = make_weight_norm_program(true);
    std::map<std::string, pnnx::MaterializedExportedTensor> state;
    state["weight_v"] = make_float_state_tensor(std::vector<int>{2, 3}, std::vector<float>{3.f, 4.f, 0.f, 0.f, 5.f, 12.f});
    state["weight_g"] = make_float_state_tensor(std::vector<int>{2, 1}, std::vector<float>{10.f, 13.f});

    pnnx::Graph graph;
    std::string error = "stale";
    const int result = pnnx::lower_exported_program(program, state, graph, error);

    check(result == 0, "weight norm", "lowering failed: " + error);
    check(error.empty(), "weight norm", "success retained an error");
    if (result != 0)
        return;

    pnnx::Operator* weight_norm = find_operator(graph, "aten::_weight_norm");
    check(weight_norm != 0, "weight norm", "missing aten::_weight_norm");
    check(weight_norm && weight_norm->inputnames == std::vector<std::string>({"v", "g", "dim"}), "weight norm", "weight_norm input names are not canonical");
    check(weight_norm && weight_norm->inputs.size() == 3, "weight norm", "weight_norm default dim was not materialized");
    if (!weight_norm || weight_norm->inputs.size() != 3)
        return;

    const pnnx::Parameter& dim = weight_norm->inputs[2]->producer->params.at("value");
    check(dim.type == 2 && dim.i == 0, "weight norm", "weight_norm default dim is wrong");

    pnnx::Operand* normalized_weight = graph.get_operand("normalized_weight");
    check(normalized_weight != 0, "weight norm", "missing normalized weight operand");

    pnnx::pass_level2(graph);
    check(count_operator(graph, "aten::_weight_norm") == 0, "weight norm pass level2", "static weight_norm was not eliminated");
    check(normalized_weight && normalized_weight->producer && normalized_weight->producer->type == "pnnx.Attribute", "weight norm pass level2", "normalized weight is not a static attribute");
    check(normalized_weight && normalized_weight->type == 1 && normalized_weight->shape == std::vector<int>({2, 3}), "weight norm pass level2", "normalized weight metadata changed");
    if (normalized_weight && normalized_weight->producer && normalized_weight->producer->type == "pnnx.Attribute" && normalized_weight->producer->has_attr("data"))
    {
        const pnnx::Attribute& data = normalized_weight->producer->attrs.at("data");
        const std::vector<float> actual = data.get_float32_data();
        const std::vector<float> expected = {6.f, 8.f, 0.f, 0.f, 5.f, 12.f};
        check(data.type == 1 && data.shape == std::vector<int>({2, 3}), "weight norm pass level2", "folded attribute metadata is wrong");
        check(actual.size() == expected.size(), "weight norm pass level2", "folded attribute element count is wrong");
        if (actual.size() == expected.size())
        {
            bool values_match = true;
            for (size_t i = 0; i < actual.size(); i++)
                values_match = values_match && fabsf(actual[i] - expected[i]) < 1e-6f;
            check(values_match, "weight norm pass level2", "folded attribute values are wrong");
        }
    }

    const pnnx::ExportedProgram dynamic_program = make_weight_norm_program(false);
    pnnx::Graph dynamic_graph;
    error = "stale";
    const int dynamic_result = pnnx::lower_exported_program(dynamic_program, std::map<std::string, pnnx::MaterializedExportedTensor>(), dynamic_graph, error);
    check(dynamic_result == 0, "dynamic weight norm", "lowering failed: " + error);
    check(error.empty(), "dynamic weight norm", "success retained an error");
    if (dynamic_result != 0)
        return;

    pnnx::pass_level2(dynamic_graph);
    pnnx::Operator* dynamic_weight_norm = find_operator(dynamic_graph, "torch._weight_norm");
    check(count_operator(dynamic_graph, "aten::_weight_norm") == 0 && dynamic_weight_norm != 0, "dynamic weight norm pass level2", "dynamic weight_norm was not canonicalized");
    check(dynamic_weight_norm && dynamic_weight_norm->inputs.size() == 2 && dynamic_weight_norm->inputnames == std::vector<std::string>({"v", "g"}), "dynamic weight norm pass level2", "dynamic weight_norm tensor inputs changed");
    check(dynamic_weight_norm && dynamic_weight_norm->has_param("dim") && dynamic_weight_norm->params.at("dim").type == 2 && dynamic_weight_norm->params.at("dim").i == 0, "dynamic weight norm pass level2", "dynamic weight_norm dim changed");
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

static void test_none_output_lowering()
{
    pnnx::ExportedProgram program = make_linear_relu_program();

    pnnx::ExportedNode item;
    item.name = "item";
    item.has_name = true;
    item.target = "torch.ops.aten.item.default";
    item.inputs.push_back(make_input("self", "relu"));
    item.outputs.push_back(pnnx::ExportedArgument());
    program.graph.nodes.push_back(item);

    pnnx::Graph graph;
    std::string error;
    const int result = pnnx::lower_exported_program(program, make_linear_state(), graph, error);

    check(result == 0, "none output", error);
    if (result != 0)
        return;

    pnnx::Operator* item_op = find_operator(graph, "aten::item");
    check(item_op != 0, "none output", "missing aten::item");
    check(item_op && item_op->inputs.size() == 1 && item_op->outputs.empty(), "none output", "None output was materialized as a tensor");
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

static void test_tensor_list_output_lowering()
{
    const pnnx::ExportedProgram program = make_chunk_program();
    pnnx::Graph graph;
    std::string error;
    const int result = pnnx::lower_exported_program(program, std::map<std::string, pnnx::MaterializedExportedTensor>(), graph, error);

    check(result == 0, "tensor list output", error);
    if (result != 0)
        return;

    pnnx::Operator* chunk = find_operator(graph, "aten::chunk");
    pnnx::Operator* unpack = find_operator(graph, "prim::ListUnpack");
    check(chunk != 0, "tensor list output", "missing aten::chunk");
    check(unpack != 0, "tensor list output", "missing prim::ListUnpack");
    check(chunk && chunk->outputs.size() == 1 && chunk->outputs[0]->consumers.size() == 1 && chunk->outputs[0]->consumers[0] == unpack, "tensor list output", "chunk list output is not connected to unpack");
    check(unpack && unpack->inputs.size() == 1 && unpack->inputs[0]->producer == chunk, "tensor list output", "unpack does not consume the chunk list");
    check(unpack && unpack->outputs.size() == 2 && unpack->outputs[0]->name == "chunk_0" && unpack->outputs[1]->name == "chunk_1", "tensor list output", "unpacked tensor names are out of order");
    check(unpack && unpack->outputs.size() == 2 && unpack->outputs[0]->shape == std::vector<int>({1, 2}) && unpack->outputs[1]->shape == std::vector<int>({1, 2}), "tensor list output", "unpacked tensor metadata is missing");

    pnnx::pass_level2(graph);
    check(count_operator(graph, "aten::chunk") == 0 && count_operator(graph, "torch.chunk") == 1, "tensor list output pass level2", "chunk was not canonicalized");
    pnnx::fuse_op1ton_unpack(graph);
    chunk = find_operator(graph, "torch.chunk");
    check(count_operator(graph, "prim::ListUnpack") == 0, "tensor list output pass level3", "list unpack was not fused");
    check(chunk && chunk->outputs.size() == 2 && chunk->outputs[0]->name == "chunk_0" && chunk->outputs[1]->name == "chunk_1", "tensor list output pass level3", "chunk outputs were not flattened in order");
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

static void test_scalar_type_argument_lowering()
{
    const int64_t exported_values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
    const int pnnx_values[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15};

    for (size_t i = 0; i < sizeof(exported_values) / sizeof(exported_values[0]); i++)
    {
        const pnnx::ExportedProgram program = make_unary_program(
            "torch.ops.aten.softmax.int", "softmax",
            std::vector<pnnx::ExportedNamedArgument>{
                make_input("dim", make_int(1)),
                make_keyword_input("dtype", make_enum(pnnx::EXPORTED_ARGUMENT_SCALAR_TYPE, exported_values[i]))});
        pnnx::Graph graph;
        std::string error;
        const int result = pnnx::lower_exported_program(program, std::map<std::string, pnnx::MaterializedExportedTensor>(), graph, error);

        check(result == 0, "scalar type enum mapping", error);
        if (result != 0)
            continue;

        pnnx::Operator* softmax = find_operator(graph, "aten::softmax");
        check(softmax && softmax->inputnames == std::vector<std::string>({"self", "dim", "dtype"}), "scalar type enum mapping", "softmax input names are not canonical");
        check(softmax && softmax->inputs.size() == 3 && softmax->inputs[2]->producer && softmax->inputs[2]->producer->has_param("value"), "scalar type enum mapping", "dtype constant is missing");
        if (softmax && softmax->inputs.size() == 3 && softmax->inputs[2]->producer && softmax->inputs[2]->producer->has_param("value"))
        {
            const pnnx::Parameter& dtype = softmax->inputs[2]->producer->params.at("value");
            check(dtype.type == 2 && dtype.i == pnnx_values[i], "scalar type enum mapping", "scalar type enum value is wrong");
        }

        pnnx::pass_level2(graph);
        check(count_operator(graph, "aten::softmax") == 0 && count_operator(graph, "F.softmax") == 1, "scalar type pass level2", "softmax did not consume the dtype constant");
    }

    const int64_t unsupported_values[] = {0, 28, 29, 32, INT64_MAX};
    for (size_t i = 0; i < sizeof(unsupported_values) / sizeof(unsupported_values[0]); i++)
    {
        const pnnx::ExportedProgram program = make_unary_program(
            "torch.ops.aten.softmax.int", "softmax",
            std::vector<pnnx::ExportedNamedArgument>{
                make_input("dim", make_int(1)),
                make_keyword_input("dtype", make_enum(pnnx::EXPORTED_ARGUMENT_SCALAR_TYPE, unsupported_values[i]))});
        pnnx::Graph graph;
        std::string error;
        const int result = pnnx::lower_exported_program(program, std::map<std::string, pnnx::MaterializedExportedTensor>(), graph, error);

        check(result != 0, "scalar type enum rejection", "unsupported scalar type unexpectedly lowered");
        check(error.find("unsupported non-tensor argument dtype") != std::string::npos, "scalar type enum rejection", "wrong error " + error);
        check(graph.ops.empty() && graph.operands.empty(), "scalar type enum rejection", "failed lowering mutated the destination graph");
    }
}

static void test_device_argument_lowering()
{
    struct DeviceCase
    {
        const char* type;
        int64_t index;
        bool has_index;
        const char* expected;
    };

    const DeviceCase cases[] = {
        {"cpu", 0, false, "cpu"},
        {"cpu", 0, true, "cpu:0"},
        {"cuda", 2, true, "cuda:2"},
    };

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
    {
        const pnnx::ExportedProgram program = make_unary_program(
            "torch.ops.aten.ones_like.default", "ones_like",
            std::vector<pnnx::ExportedNamedArgument>{
                make_keyword_input("device", make_device(cases[i].type, cases[i].index, cases[i].has_index))});
        pnnx::Graph graph;
        std::string error;
        const int result = pnnx::lower_exported_program(program, std::map<std::string, pnnx::MaterializedExportedTensor>(), graph, error);

        check(result == 0, "device argument", error);
        if (result != 0)
            continue;

        pnnx::Operator* ones_like = find_operator(graph, "aten::ones_like");
        check(ones_like && ones_like->inputnames == std::vector<std::string>({"self", "dtype", "layout", "device", "pin_memory", "memory_format"}), "device argument", "ones_like input names are not canonical");
        check(ones_like && ones_like->inputs.size() == 6 && ones_like->inputs[3]->producer && ones_like->inputs[3]->producer->has_param("value"), "device argument", "device constant is missing");
        if (ones_like && ones_like->inputs.size() == 6 && ones_like->inputs[3]->producer && ones_like->inputs[3]->producer->has_param("value"))
        {
            const pnnx::Parameter& device = ones_like->inputs[3]->producer->params.at("value");
            check(device.type == 4 && device.s == cases[i].expected, "device argument", "device string is wrong");
        }

        pnnx::pass_level2(graph);
        check(count_operator(graph, "aten::ones_like") == 0 && count_operator(graph, "torch.ones_like") == 1, "device argument pass level2", "ones_like did not consume the device constant");
    }

    const DeviceCase unsupported_cases[] = {
        {"", 0, false, ""},
        {"cuda", -1, true, ""},
        {"cuda", 128, true, ""},
    };

    for (size_t i = 0; i < sizeof(unsupported_cases) / sizeof(unsupported_cases[0]); i++)
    {
        const pnnx::ExportedProgram program = make_unary_program(
            "torch.ops.aten.ones_like.default", "ones_like",
            std::vector<pnnx::ExportedNamedArgument>{
                make_keyword_input("device", make_device(unsupported_cases[i].type, unsupported_cases[i].index, unsupported_cases[i].has_index))});
        pnnx::Graph graph;
        std::string error;
        const int result = pnnx::lower_exported_program(program, std::map<std::string, pnnx::MaterializedExportedTensor>(), graph, error);

        check(result != 0, "device argument rejection", "invalid device unexpectedly lowered");
        check(error.find("unsupported non-tensor argument device") != std::string::npos, "device argument rejection", "wrong error " + error);
        check(graph.ops.empty() && graph.operands.empty(), "device argument rejection", "failed lowering mutated the destination graph");
    }
}

static void test_layout_argument_lowering()
{
    {
        const pnnx::ExportedProgram program = make_unary_program(
            "torch.ops.aten.ones_like.default", "ones_like",
            std::vector<pnnx::ExportedNamedArgument>{
                make_keyword_input("layout", make_enum(pnnx::EXPORTED_ARGUMENT_LAYOUT, 7))});
        pnnx::Graph graph;
        std::string error;
        const int result = pnnx::lower_exported_program(program, std::map<std::string, pnnx::MaterializedExportedTensor>(), graph, error);

        check(result == 0, "layout argument", error);
        if (result == 0)
        {
            pnnx::Operator* ones_like = find_operator(graph, "aten::ones_like");
            check(ones_like && ones_like->inputnames == std::vector<std::string>({"self", "dtype", "layout", "device", "pin_memory", "memory_format"}), "layout argument", "ones_like input names are not canonical");
            check(ones_like && ones_like->inputs.size() == 6 && ones_like->inputs[2]->producer && ones_like->inputs[2]->producer->has_param("value"), "layout argument", "layout constant is missing");
            if (ones_like && ones_like->inputs.size() == 6 && ones_like->inputs[2]->producer && ones_like->inputs[2]->producer->has_param("value"))
            {
                const pnnx::Parameter& layout = ones_like->inputs[2]->producer->params.at("value");
                check(layout.type == 2 && layout.i == 0, "layout argument", "strided layout was not converted to the pnnx/JIT enum value");
            }

            pnnx::pass_level2(graph);
            check(count_operator(graph, "aten::ones_like") == 0 && count_operator(graph, "torch.ones_like") == 1, "layout argument pass level2", "ones_like did not consume the layout constant");
        }
    }

    const int64_t unsupported_values[] = {0, 1, 2, 3, 4, 5, 6, 8, INT64_MAX};
    for (size_t i = 0; i < sizeof(unsupported_values) / sizeof(unsupported_values[0]); i++)
    {
        const pnnx::ExportedProgram program = make_unary_program(
            "torch.ops.aten.ones_like.default", "ones_like",
            std::vector<pnnx::ExportedNamedArgument>{
                make_keyword_input("layout", make_enum(pnnx::EXPORTED_ARGUMENT_LAYOUT, unsupported_values[i]))});
        pnnx::Graph graph;
        std::string error;
        const int result = pnnx::lower_exported_program(program, std::map<std::string, pnnx::MaterializedExportedTensor>(), graph, error);

        check(result != 0, "layout argument rejection", "unsupported layout unexpectedly lowered");
        check(error.find("unsupported non-tensor argument layout") != std::string::npos, "layout argument rejection", "wrong error " + error);
        check(graph.ops.empty() && graph.operands.empty(), "layout argument rejection", "failed lowering mutated the destination graph");
    }
}

static void test_float_list_argument_lowering()
{
    {
        pnnx::ExportedProgram program = make_unary_program(
            "torch.ops.aten.upsample_nearest2d.vec", "upsample_nearest2d",
            std::vector<pnnx::ExportedNamedArgument>{
                make_input("output_size", pnnx::ExportedArgument()),
                make_input("scale_factors", make_floats(std::vector<double>{2.0, 2.976744}))});
        program.graph.nodes[0].inputs[0].name = "input";
        pnnx::Graph graph;
        std::string error;
        const int result = pnnx::lower_exported_program(program, std::map<std::string, pnnx::MaterializedExportedTensor>(), graph, error);

        check(result == 0, "float list argument", error);
        if (result == 0)
        {
            pnnx::Operator* upsample = find_operator(graph, "aten::upsample_nearest2d");
            check(upsample && upsample->inputnames == std::vector<std::string>({"input", "output_size", "scale_factors"}), "float list argument", "upsample input names are not canonical");
            check(upsample && upsample->inputs.size() == 3 && upsample->inputs[2]->producer && upsample->inputs[2]->producer->has_param("value"), "float list argument", "scale_factors constant is missing");
            if (upsample && upsample->inputs.size() == 3 && upsample->inputs[2]->producer && upsample->inputs[2]->producer->has_param("value"))
            {
                const pnnx::Parameter& scale_factors = upsample->inputs[2]->producer->params.at("value");
                check(scale_factors.type == 6 && scale_factors.af == std::vector<float>({2.f, 2.976744f}), "float list argument", "scale_factors list changed");
            }

            pnnx::pass_level2(graph);
            pnnx::Operator* functional_upsample = find_operator(graph, "F.upsample_nearest");
            check(count_operator(graph, "aten::upsample_nearest2d") == 0 && functional_upsample != 0, "float list argument pass level2", "upsample was not canonicalized");
            check(functional_upsample && functional_upsample->inputs.size() == 2 && functional_upsample->inputs[1]->producer && functional_upsample->inputs[1]->producer->has_param("value") && functional_upsample->inputs[1]->producer->params.at("value").type == 6 && functional_upsample->inputs[1]->producer->params.at("value").af == std::vector<float>({2.f, 2.976744f}), "float list argument pass level2", "canonical upsample lost scale_factors");
        }
    }

    {
        pnnx::ExportedProgram program = make_unary_program(
            "torch.ops.aten.upsample_nearest2d.vec", "upsample_nearest2d",
            std::vector<pnnx::ExportedNamedArgument>{
                make_input("output_size", pnnx::ExportedArgument()),
                make_input("scale_factors", make_floats(std::vector<double>()))});
        program.graph.nodes[0].inputs[0].name = "input";
        pnnx::Graph graph;
        std::string error;
        const int result = pnnx::lower_exported_program(program, std::map<std::string, pnnx::MaterializedExportedTensor>(), graph, error);

        check(result == 0, "empty float list argument", error);
        if (result == 0)
        {
            pnnx::Operator* upsample = find_operator(graph, "aten::upsample_nearest2d");
            check(upsample && upsample->inputs.size() == 3 && upsample->inputs[2]->producer && upsample->inputs[2]->producer->has_param("value"), "empty float list argument", "empty scale_factors constant is missing");
            if (upsample && upsample->inputs.size() == 3 && upsample->inputs[2]->producer && upsample->inputs[2]->producer->has_param("value"))
            {
                const pnnx::Parameter& scale_factors = upsample->inputs[2]->producer->params.at("value");
                check(scale_factors.type == 6 && scale_factors.af.empty(), "empty float list argument", "empty scale_factors list changed");
            }
        }
    }
}

static void test_empty_int_list_parameter_roundtrip()
{
    const pnnx::Parameter parsed = pnnx::Parameter::parse_from_string("[]");
    check(parsed.type == 5 && parsed.ai.empty(), "empty int list parameter", "[] was not parsed as an empty integer list");

    const pnnx::Parameter source = std::vector<int>();
    const std::string encoded = pnnx::Parameter::encode_to_string(source);
    check(encoded == "[]", "empty int list parameter", "empty integer list was not encoded distinctly from None");

    const pnnx::Parameter roundtrip = pnnx::Parameter::parse_from_string(encoded);
    check(roundtrip.type == 5 && roundtrip.ai.empty(), "empty int list parameter", "empty integer list changed during text roundtrip");
}

static void test_nearest_exact_vec_size_lowering()
{
    struct NearestExactCase
    {
        const char* target;
        const char* aten_type;
        const char* output_name;
        std::vector<int64_t> output_size;
    };

    const NearestExactCase cases[] = {
        {"torch.ops.aten._upsample_nearest_exact2d.vec", "aten::_upsample_nearest_exact2d", "upsample_nearest_exact2d", std::vector<int64_t>{11, 12}},
        {"torch.ops.aten._upsample_nearest_exact3d.vec", "aten::_upsample_nearest_exact3d", "upsample_nearest_exact3d", std::vector<int64_t>{11, 12, 13}},
    };

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
    {
        const NearestExactCase& c = cases[i];
        pnnx::ExportedProgram program = make_unary_program(
            c.target, c.output_name,
            std::vector<pnnx::ExportedNamedArgument>{
                make_input("output_size", make_ints(c.output_size)),
                make_input("scale_factors", pnnx::ExportedArgument())});
        program.graph.nodes[0].inputs[0].name = "input";

        pnnx::Graph graph;
        std::string error;
        const int result = pnnx::lower_exported_program(program, std::map<std::string, pnnx::MaterializedExportedTensor>(), graph, error);

        check(result == 0, "nearest exact vec size", error);
        if (result == 0)
        {
            check(find_operator(graph, c.aten_type) != 0, "nearest exact vec size", "front end did not preserve the aten operator");

            pnnx::pass_level2(graph);
            pnnx::Operator* interpolate = find_operator(graph, "F.interpolate");
            check(find_operator(graph, c.aten_type) == 0 && interpolate != 0, "nearest exact vec size pass level2", "nearest-exact operator was not canonicalized");
            check(interpolate && interpolate->has_param("size") && interpolate->params.at("size").type == 5 && interpolate->params.at("size").ai == std::vector<int>(c.output_size.begin(), c.output_size.end()), "nearest exact vec size pass level2", "output size changed");
            check(interpolate && interpolate->has_param("mode") && interpolate->params.at("mode").type == 4 && interpolate->params.at("mode").s == "nearest-exact", "nearest exact vec size pass level2", "interpolation mode changed");
            check(interpolate && interpolate->has_param("recompute_scale_factor") && interpolate->params.at("recompute_scale_factor").type == 1 && !interpolate->params.at("recompute_scale_factor").b, "nearest exact vec size pass level2", "recompute_scale_factor changed");
        }
    }
}

static void test_avg_pool_empty_stride_normalization()
{
    struct AvgPoolCase
    {
        const char* target;
        const char* aten_type;
        const char* functional_type;
        const char* output_name;
        std::vector<int64_t> kernel_size;
        std::vector<int64_t> stride;
        std::vector<int64_t> padding;
        bool has_divisor_override;
    };

    const AvgPoolCase cases[] = {
        {"torch.ops.aten.avg_pool1d.default", "aten::avg_pool1d", "F.avg_pool1d", "avg_pool1d", std::vector<int64_t>{2}, std::vector<int64_t>{1}, std::vector<int64_t>{0}, false},
        {"torch.ops.aten.avg_pool2d.default", "aten::avg_pool2d", "F.avg_pool2d", "avg_pool2d", std::vector<int64_t>{2, 2}, std::vector<int64_t>{1, 2}, std::vector<int64_t>{0, 0}, true},
        {"torch.ops.aten.avg_pool3d.default", "aten::avg_pool3d", "F.avg_pool3d", "avg_pool3d", std::vector<int64_t>{2, 2, 2}, std::vector<int64_t>{1, 2, 1}, std::vector<int64_t>{0, 0, 0}, true},
    };

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
    {
        const AvgPoolCase& c = cases[i];
        std::vector<pnnx::ExportedNamedArgument> arguments;
        arguments.push_back(make_input("kernel_size", make_ints(c.kernel_size)));
        arguments.push_back(make_input("stride", make_ints(std::vector<int64_t>())));
        arguments.push_back(make_input("padding", make_ints(c.padding)));
        arguments.push_back(make_input("ceil_mode", make_bool(false)));
        arguments.push_back(make_input("count_include_pad", make_bool(true)));
        if (c.has_divisor_override)
            arguments.push_back(make_input("divisor_override", pnnx::ExportedArgument()));

        pnnx::ExportedProgram program = make_unary_program(c.target, c.output_name, arguments);
        pnnx::Graph graph;
        std::string error;
        const int result = pnnx::lower_exported_program(program, std::map<std::string, pnnx::MaterializedExportedTensor>(), graph, error);

        check(result == 0, "avg pool empty stride", error);
        if (result == 0)
        {
            pnnx::Operator* avg_pool = find_operator(graph, c.aten_type);
            check(avg_pool && avg_pool->inputs.size() >= 3 && avg_pool->inputs[2]->producer && avg_pool->inputs[2]->producer->has_param("value"), "avg pool empty stride", "raw stride constant is missing");
            if (avg_pool && avg_pool->inputs.size() >= 3 && avg_pool->inputs[2]->producer && avg_pool->inputs[2]->producer->has_param("value"))
            {
                const pnnx::Parameter& stride = avg_pool->inputs[2]->producer->params.at("value");
                check(stride.type == 5 && stride.ai.empty(), "avg pool empty stride", "front end changed the serialized empty list");
            }

            pnnx::pass_level2(graph);
            pnnx::Operator* functional_avg_pool = find_operator(graph, c.functional_type);
            check(functional_avg_pool != 0, "avg pool empty stride pass level2", "avg pool was not canonicalized");
            check(functional_avg_pool && functional_avg_pool->has_param("stride") && functional_avg_pool->params.at("stride").type == 0, "avg pool empty stride pass level2", "empty stride was not normalized to None");
        }

        arguments[1] = make_input("stride", make_ints(c.stride));
        program = make_unary_program(c.target, c.output_name, arguments);
        pnnx::Graph nonempty_graph;
        error.clear();
        const int nonempty_result = pnnx::lower_exported_program(program, std::map<std::string, pnnx::MaterializedExportedTensor>(), nonempty_graph, error);

        check(nonempty_result == 0, "avg pool nonempty stride", error);
        if (nonempty_result == 0)
        {
            pnnx::pass_level2(nonempty_graph);
            pnnx::Operator* functional_avg_pool = find_operator(nonempty_graph, c.functional_type);
            check(functional_avg_pool && functional_avg_pool->has_param("stride") && functional_avg_pool->params.at("stride").type == 5 && functional_avg_pool->params.at("stride").ai == std::vector<int>(c.stride.begin(), c.stride.end()), "avg pool nonempty stride pass level2", "explicit stride changed");
        }
    }
}

static void test_max_pool_empty_stride_normalization()
{
    struct MaxPoolCase
    {
        const char* target;
        const char* aten_type;
        const char* functional_type;
        const char* output_name;
        std::vector<int64_t> kernel_size;
        std::vector<int64_t> stride;
        std::vector<int64_t> padding;
        std::vector<int64_t> dilation;
    };

    const MaxPoolCase cases[] = {
        {"torch.ops.aten.max_pool1d.default", "aten::max_pool1d", "F.max_pool1d", "max_pool1d", std::vector<int64_t>{2}, std::vector<int64_t>{1}, std::vector<int64_t>{0}, std::vector<int64_t>{1}},
        {"torch.ops.aten.max_pool2d.default", "aten::max_pool2d", "F.max_pool2d", "max_pool2d", std::vector<int64_t>{2, 2}, std::vector<int64_t>{1, 2}, std::vector<int64_t>{0, 0}, std::vector<int64_t>{1, 1}},
        {"torch.ops.aten.max_pool3d.default", "aten::max_pool3d", "F.max_pool3d", "max_pool3d", std::vector<int64_t>{2, 2, 2}, std::vector<int64_t>{1, 2, 1}, std::vector<int64_t>{0, 0, 0}, std::vector<int64_t>{1, 1, 1}},
    };

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
    {
        const MaxPoolCase& c = cases[i];
        std::vector<pnnx::ExportedNamedArgument> arguments;
        arguments.push_back(make_input("kernel_size", make_ints(c.kernel_size)));
        arguments.push_back(make_input("stride", make_ints(std::vector<int64_t>())));
        arguments.push_back(make_input("padding", make_ints(c.padding)));
        arguments.push_back(make_input("dilation", make_ints(c.dilation)));
        arguments.push_back(make_input("ceil_mode", make_bool(false)));

        pnnx::ExportedProgram program = make_unary_program(c.target, c.output_name, arguments);
        pnnx::Graph graph;
        std::string error;
        const int result = pnnx::lower_exported_program(program, std::map<std::string, pnnx::MaterializedExportedTensor>(), graph, error);

        check(result == 0, "max pool empty stride", error);
        if (result == 0)
        {
            pnnx::Operator* max_pool = find_operator(graph, c.aten_type);
            check(max_pool && max_pool->inputs.size() >= 3 && max_pool->inputs[2]->producer && max_pool->inputs[2]->producer->has_param("value"), "max pool empty stride", "raw stride constant is missing");
            if (max_pool && max_pool->inputs.size() >= 3 && max_pool->inputs[2]->producer && max_pool->inputs[2]->producer->has_param("value"))
            {
                const pnnx::Parameter& stride = max_pool->inputs[2]->producer->params.at("value");
                check(stride.type == 5 && stride.ai.empty(), "max pool empty stride", "front end changed the serialized empty list");
            }

            pnnx::pass_level2(graph);
            pnnx::Operator* functional_max_pool = find_operator(graph, c.functional_type);
            check(functional_max_pool != 0, "max pool empty stride pass level2", "max pool was not canonicalized");
            check(functional_max_pool && functional_max_pool->has_param("stride") && functional_max_pool->params.at("stride").type == 0, "max pool empty stride pass level2", "empty stride was not normalized to None");
        }

        arguments[1] = make_input("stride", make_ints(c.stride));
        program = make_unary_program(c.target, c.output_name, arguments);
        pnnx::Graph nonempty_graph;
        error.clear();
        const int nonempty_result = pnnx::lower_exported_program(program, std::map<std::string, pnnx::MaterializedExportedTensor>(), nonempty_graph, error);

        check(nonempty_result == 0, "max pool nonempty stride", error);
        if (nonempty_result == 0)
        {
            pnnx::pass_level2(nonempty_graph);
            pnnx::Operator* functional_max_pool = find_operator(nonempty_graph, c.functional_type);
            check(functional_max_pool && functional_max_pool->has_param("stride") && functional_max_pool->params.at("stride").type == 5 && functional_max_pool->params.at("stride").ai == std::vector<int>(c.stride.begin(), c.stride.end()), "max pool nonempty stride pass level2", "explicit stride changed");
        }
    }
}

static void test_alias_elimination()
{
    {
        const pnnx::ExportedProgram program = make_unary_program("torch.ops.aten.alias.default", "alias", std::vector<pnnx::ExportedNamedArgument>());
        pnnx::Graph graph;
        std::string error;
        const int result = pnnx::lower_exported_program(program, std::map<std::string, pnnx::MaterializedExportedTensor>(), graph, error);

        check(result == 0, "alias elimination", error);
        if (result == 0)
        {
            check(count_operator(graph, "aten::alias") == 1, "alias elimination", "missing imported alias");
            pnnx::pass_level2(graph);
            check(count_operator(graph, "aten::alias") == 0, "alias elimination", "alias was not eliminated");

            pnnx::Operator* input = find_operator(graph, "pnnx.Input");
            pnnx::Operator* output = find_operator(graph, "pnnx.Output");
            check(input && output && input->outputs.size() == 1 && output->inputs.size() == 1 && input->outputs[0] == output->inputs[0], "alias elimination", "alias consumers were not rewired to the source tensor");
        }
    }

    {
        pnnx::Graph graph;
        pnnx::Operator* input = graph.new_operator("pnnx.Input", "input");
        pnnx::Operand* x = graph.new_operand("x");
        x->producer = input;
        input->outputs.push_back(x);

        pnnx::Operator* rhs_input = graph.new_operator("pnnx.Input", "rhs_input");
        pnnx::Operand* rhs = graph.new_operand("rhs");
        rhs->producer = rhs_input;
        rhs_input->outputs.push_back(rhs);

        pnnx::Operator* alias = graph.new_operator("aten::alias", "alias");
        alias->inputs.push_back(x);
        x->consumers.push_back(alias);
        pnnx::Operand* y = graph.new_operand("y");
        y->producer = alias;
        alias->outputs.push_back(y);

        pnnx::Operator* add = graph.new_operator("aten::add_", "add_");
        add->inputs.push_back(y);
        add->inputs.push_back(rhs);
        y->consumers.push_back(add);
        rhs->consumers.push_back(add);
        pnnx::Operand* sum = graph.new_operand("sum");
        sum->producer = add;
        add->outputs.push_back(sum);

        pnnx::Operator* output = graph.new_operator("pnnx.Output", "output");
        output->inputs.push_back(x);
        x->consumers.push_back(output);

        pnnx::pass_level2(graph);

        check(count_operator(graph, "aten::alias") == 0, "alias functionize", "alias was not eliminated after functionize");
        check(count_operator(graph, "aten::add_") == 0 && count_operator(graph, "aten::add") == 1, "alias functionize", "in-place operator was not functionized");
        pnnx::Operator* copy = find_operator(graph, "aten::copy");
        output = find_operator(graph, "pnnx.Output");
        check(copy && output && copy->outputs.size() == 1 && output->inputs.size() == 1 && output->inputs[0] == copy->outputs[0], "alias functionize", "mutation through alias was not propagated to the source tensor");
    }
}

static void test_lift_fresh_copy_lowering()
{
    const pnnx::ExportedProgram program = make_unary_program("torch.ops.aten.lift_fresh_copy.default", "lift_fresh_copy", std::vector<pnnx::ExportedNamedArgument>());
    pnnx::Graph graph;
    std::string error;
    const int result = pnnx::lower_exported_program(program, std::map<std::string, pnnx::MaterializedExportedTensor>(), graph, error);

    check(result == 0, "lift fresh copy lowering", error);
    if (result == 0)
    {
        check(count_operator(graph, "aten::lift_fresh_copy") == 1, "lift fresh copy lowering", "missing imported lift_fresh_copy");
        pnnx::pass_level2(graph);
        check(count_operator(graph, "aten::lift_fresh_copy") == 0, "lift fresh copy lowering", "lift_fresh_copy was not lowered");

        pnnx::Operator* clone = find_operator(graph, "torch.clone");
        check(clone && clone->has_param("memory_format") && clone->params.at("memory_format").s == "torch.contiguous_format", "lift fresh copy lowering", "lift_fresh_copy was not lowered to a contiguous fresh clone");

        pnnx::Operator* input = find_operator(graph, "pnnx.Input");
        pnnx::Operator* output = find_operator(graph, "pnnx.Output");
        check(input && clone && output && input->outputs.size() == 1 && clone->inputs.size() == 1 && clone->outputs.size() == 1 && output->inputs.size() == 1 && clone->inputs[0] == input->outputs[0] && output->inputs[0] == clone->outputs[0], "lift fresh copy lowering", "clone wiring does not preserve the fresh-copy dataflow");
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

        pnnx::Graph graph;
        std::string error = "stale";
        const int result = pnnx::lower_exported_program(program, make_linear_state(), graph, error);
        check(result == 0, "non-persistent buffer", "lowering failed: " + error);
        check(error.empty(), "non-persistent buffer", "success retained an error");
        check(count_operator(graph, "pnnx.Attribute") == 2, "non-persistent buffer", "buffer was not materialized as a static attribute");
        check(count_operator(graph, "pnnx.Input") == 1, "non-persistent buffer", "buffer became a runtime input");
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
        expect_lower_error(program, make_linear_state(), "tensor x has a negative size at dimension 0", "negative tensor size");
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

    {
        pnnx::ExportedProgram program = make_chunk_program();
        program.graph.tensor_values.erase("chunk_1");
        expect_lower_error(program, std::map<std::string, pnnx::MaterializedExportedTensor>(), "missing tensor metadata for chunk_1", "missing tensor-list output metadata");
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
            if (meta_it->second.sizes[j].type != pnnx::EXPORTED_SYM_INT_STATIC || meta_it->second.sizes[j].value < 0 || meta_it->second.sizes[j].value > INT_MAX)
            {
                fprintf(stderr, "real linear fixture shape is out of range\n");
                return 1;
            }
            tensor.shape.push_back((int)meta_it->second.sizes[j].value);
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
    test_bounded_dynamic_result_shape();
    test_higher_order_graph_lowering();
    test_complex_scalar_lowering();
    test_output_tree_lowering();
    test_linear_default_bias_constant();
    test_conv2d_constant_arguments();
    test_conv_nd_direct_lowering();
    test_conv_transpose_nd_direct_lowering();
    test_batch_norm_constant_arguments();
    test_instance_norm_running_stats();
    test_weight_norm_lowering();
    test_inplace_relu_is_functionalized_by_level2();
    test_max_pool2d_constant_arguments();
    test_inplace_add_is_functionalized_by_level2();
    test_adaptive_avg_pool2d_output_size();
    test_flatten_dimensions();
    test_dispatcher_backed_target_lowering();
    test_none_output_lowering();
    test_tensor_list_argument_lowering();
    test_tensor_list_output_lowering();
    test_string_and_memory_format_argument_lowering();
    test_scalar_type_argument_lowering();
    test_device_argument_lowering();
    test_layout_argument_lowering();
    test_float_list_argument_lowering();
    test_empty_int_list_parameter_roundtrip();
    test_nearest_exact_vec_size_lowering();
    test_avg_pool_empty_stride_normalization();
    test_max_pool_empty_stride_normalization();
    test_alias_elimination();
    test_lift_fresh_copy_lowering();
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
