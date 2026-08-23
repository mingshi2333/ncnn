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
        program.graph.nodes[1].target = "torch.ops.aten.sigmoid.default";
        expect_lower_error(program, make_linear_state(), "unsupported exported operator torch.ops.aten.sigmoid.default", "unsupported operator target");
    }

    {
        pnnx::ExportedProgram program = make_linear_relu_program();
        program.graph.nodes[1].target = "aten::relu";
        expect_lower_error(program, make_linear_state(), "torch 2.12.1+cu126 aten opset 10 target aten::relu", "malformed operator context");
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
