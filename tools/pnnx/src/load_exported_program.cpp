// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "load_exported_program.h"

#include "exported_program_graph.h"
#include "exported_program_operator.h"
#include "pt2_archive.h"

#include <limits.h>

#include <complex>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace pnnx {

static bool is_symbol_name(const std::string& value)
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

static bool parse_plain_integer_symbol(const std::string& expression, std::string& symbol)
{
    const std::string prefix = "Symbol('";
    const std::string suffix = "', integer=True)";
    if (expression.size() <= prefix.size() + suffix.size())
        return false;
    if (expression.compare(0, prefix.size(), prefix) != 0)
        return false;
    if (expression.compare(expression.size() - suffix.size(), suffix.size(), suffix) != 0)
        return false;

    symbol = expression.substr(prefix.size(), expression.size() - prefix.size() - suffix.size());
    return is_symbol_name(symbol);
}

static int resolve_tensor_size(const ExportedProgram& program, const std::string& name, size_t dimension, const ExportedSymInt& size, int& resolved, std::string& error)
{
    if (size.type == EXPORTED_SYM_INT_STATIC)
    {
        if (size.value < 0)
        {
            std::ostringstream message;
            message << "tensor " << name << " has a negative size at dimension " << dimension;
            error = message.str();
            return -1;
        }
        if (size.value > INT_MAX)
        {
            error = "tensor shape does not fit pnnx for " + name;
            return -1;
        }

        resolved = (int)size.value;
        return 0;
    }

    std::string symbol;
    if (!parse_plain_integer_symbol(size.expression, symbol))
    {
        std::ostringstream message;
        message << "tensor " << name << " has unsupported symbolic size " << size.expression << " at dimension " << dimension;
        error = message.str();
        return -1;
    }

    const std::map<std::string, ExportedRangeConstraint>::const_iterator constraint_it = program.range_constraints.find(symbol);
    if (constraint_it == program.range_constraints.end() || !constraint_it->second.has_max)
    {
        std::ostringstream message;
        message << "tensor " << name << " symbolic size " << symbol << " has no finite upper bound at dimension " << dimension;
        error = message.str();
        return -1;
    }
    if (constraint_it->second.max < 0 || constraint_it->second.max > INT_MAX)
    {
        std::ostringstream message;
        message << "tensor " << name << " symbolic size " << symbol << " upper bound does not fit pnnx at dimension " << dimension;
        error = message.str();
        return -1;
    }

    resolved = (int)constraint_it->second.max;
    return 0;
}

static int set_tensor_metadata(const ExportedProgram& program, const std::string& name, Operand* operand, std::string& error)
{
    const std::map<std::string, ExportedTensorMeta>::const_iterator meta_it = program.graph.tensor_values.find(name);
    if (meta_it == program.graph.tensor_values.end())
    {
        error = "missing tensor metadata for " + name;
        return -1;
    }

    const ExportedTensorMeta& meta = meta_it->second;
    if (meta.layout != 7)
    {
        std::ostringstream message;
        message << "unsupported tensor layout " << meta.layout << " for " << name;
        error = message.str();
        return -1;
    }
    const int pnnx_type = exported_tensor_dtype_to_pnnx_type(meta.dtype);
    if (pnnx_type == 0)
    {
        std::ostringstream message;
        message << "unsupported exported tensor dtype " << meta.dtype << " for " << name;
        error = message.str();
        return -1;
    }

    std::vector<int> shape;
    shape.reserve(meta.sizes.size());
    for (size_t i = 0; i < meta.sizes.size(); i++)
    {
        int resolved = 0;
        if (resolve_tensor_size(program, name, i, meta.sizes[i], resolved, error) != 0)
            return -1;
        shape.push_back(resolved);
    }

    operand->type = pnnx_type;
    operand->shape.swap(shape);
    return 0;
}

static const char* exported_state_kind_name(ExportedInputKind kind)
{
    if (kind == EXPORTED_PARAMETER)
        return "parameter";
    if (kind == EXPORTED_BUFFER)
        return "buffer";
    return "tensor constant";
}

static const char* exported_output_kind_name(ExportedOutputKind kind)
{
    if (kind == EXPORTED_LOSS_OUTPUT)
        return "loss output";
    if (kind == EXPORTED_BUFFER_MUTATION)
        return "buffer mutation";
    if (kind == EXPORTED_PARAMETER_MUTATION)
        return "parameter mutation";
    if (kind == EXPORTED_GRADIENT_TO_PARAMETER)
        return "gradient to parameter";
    if (kind == EXPORTED_GRADIENT_TO_USER_INPUT)
        return "gradient to user input";
    if (kind == EXPORTED_USER_INPUT_MUTATION)
        return "user input mutation";
    if (kind == EXPORTED_OUTPUT_TOKEN)
        return "output token";

    return "unknown output kind";
}

static int validate_signature_kinds(const ExportedProgram& program, std::string& error)
{
    for (size_t i = 0; i < program.input_specs.size(); i++)
    {
        const ExportedInputSpec& spec = program.input_specs[i];
        if (spec.kind == EXPORTED_USER_INPUT)
        {
            if (spec.arg.type != EXPORTED_ARGUMENT_TENSOR)
            {
                error = "user input " + spec.arg.name + " must be a tensor";
                return -1;
            }
            continue;
        }
        if (spec.kind == EXPORTED_PARAMETER || spec.kind == EXPORTED_TENSOR_CONSTANT)
        {
            if (spec.arg.type != EXPORTED_ARGUMENT_TENSOR)
            {
                error = "state input " + spec.target + " must be a tensor";
                return -1;
            }
            continue;
        }
        if (spec.kind == EXPORTED_BUFFER)
        {
            if (spec.arg.type != EXPORTED_ARGUMENT_TENSOR)
            {
                error = "buffer " + spec.target + " must be a tensor";
                return -1;
            }
            continue;
        }
        if (spec.kind == EXPORTED_CUSTOM_OBJ)
        {
            error = "custom object input " + spec.arg.name + " is unsupported";
            return -1;
        }
        if (spec.kind == EXPORTED_TOKEN)
        {
            error = "token input " + spec.arg.name + " is unsupported";
            return -1;
        }

        error = "constant input " + spec.arg.name + " is unsupported";
        return -1;
    }

    for (size_t i = 0; i < program.output_specs.size(); i++)
    {
        const ExportedOutputSpec& spec = program.output_specs[i];
        if (spec.kind != EXPORTED_USER_OUTPUT)
        {
            error = std::string("unsupported exported program ") + exported_output_kind_name(spec.kind);
            return -1;
        }
        if (spec.arg.type != EXPORTED_ARGUMENT_TENSOR)
        {
            error = "user output " + spec.arg.name + " must be a tensor";
            return -1;
        }
    }

    return 0;
}

static int validate_signature_arguments(const ExportedProgram& program, std::string& error)
{
    for (size_t i = 0; i < program.input_specs.size(); i++)
    {
        const ExportedArgument& signature = program.input_specs[i].arg;
        const ExportedArgument& graph = program.graph.inputs[i];
        if (signature.type != graph.type || signature.name != graph.name)
        {
            std::ostringstream message;
            message << "input spec " << i << " tensor " << signature.name << " does not match graph input " << graph.name;
            error = message.str();
            return -1;
        }
    }

    for (size_t i = 0; i < program.output_specs.size(); i++)
    {
        const ExportedArgument& signature = program.output_specs[i].arg;
        const ExportedArgument& graph = program.graph.outputs[i];
        if (signature.type != graph.type || signature.name != graph.name)
        {
            std::ostringstream message;
            message << "output spec " << i << " tensor " << signature.name << " does not match graph output " << graph.name;
            error = message.str();
            return -1;
        }
    }

    return 0;
}

static std::string unique_name(const std::string& requested, std::set<std::string>& names)
{
    if (names.insert(requested).second)
        return requested;

    for (size_t suffix = 1;; suffix++)
    {
        std::ostringstream candidate;
        candidate << requested << '_' << suffix;
        if (names.insert(candidate.str()).second)
            return candidate.str();
    }
}

static int validate_output_tree(const ExportedTreeSpec& tree_spec, size_t& leaf_count, std::string& error)
{
    if (tree_spec.type == EXPORTED_TREE_SPEC_LEAF)
    {
        if (!tree_spec.children.empty())
        {
            error = "output treespec leaf must not have children";
            return -1;
        }

        leaf_count++;
        return 0;
    }

    if (tree_spec.type != EXPORTED_TREE_SPEC_TUPLE && tree_spec.type != EXPORTED_TREE_SPEC_LIST)
    {
        error = "invalid output treespec type";
        return -1;
    }

    for (size_t i = 0; i < tree_spec.children.size(); i++)
    {
        if (validate_output_tree(tree_spec.children[i], leaf_count, error) != 0)
            return -1;
    }

    return 0;
}

static int construct_output_tree(const ExportedTreeSpec& tree_spec,
                                 const std::vector<Operand*>& flat_outputs,
                                 size_t& flat_index,
                                 Graph& graph,
                                 std::set<std::string>& operand_names,
                                 std::set<std::string>& operator_names,
                                 int& unknown_index,
                                 Operand*& output,
                                 std::string& error)
{
    if (tree_spec.type == EXPORTED_TREE_SPEC_LEAF)
    {
        if (flat_index >= flat_outputs.size())
        {
            error = "output treespec consumes too many graph outputs";
            return -1;
        }

        output = flat_outputs[flat_index++];
        return 0;
    }

    std::vector<Operand*> children;
    children.reserve(tree_spec.children.size());
    for (size_t i = 0; i < tree_spec.children.size(); i++)
    {
        Operand* child = 0;
        if (construct_output_tree(tree_spec.children[i], flat_outputs, flat_index, graph, operand_names, operator_names, unknown_index, child, error) != 0)
            return -1;
        children.push_back(child);
    }

    std::ostringstream generated_name;
    generated_name << "pnnx_" << unknown_index++;
    const char* operator_type = tree_spec.type == EXPORTED_TREE_SPEC_TUPLE ? "prim::TupleConstruct" : "prim::ListConstruct";
    Operator* construct = graph.new_operator(operator_type, unique_name(generated_name.str(), operator_names));
    output = graph.new_operand(unique_name(generated_name.str(), operand_names));
    output->producer = construct;
    construct->outputs.push_back(output);
    for (size_t i = 0; i < children.size(); i++)
    {
        children[i]->consumers.push_back(construct);
        construct->inputs.push_back(children[i]);
    }

    return 0;
}

static int exported_int_to_pnnx(int64_t value)
{
    if (value == std::numeric_limits<int64_t>::max()) value = INT_MAX;
    if (value == std::numeric_limits<int64_t>::max() - 1) value = INT_MAX - 1;
    if (value == std::numeric_limits<int64_t>::min()) value = INT_MIN;
    if (value == std::numeric_limits<int64_t>::min() + 1) value = INT_MIN + 1;
    return (int)value;
}

static int exported_memory_format_to_pnnx(int64_t value, int& converted)
{
    if (value == 1)
        converted = 0; // contiguous
    else if (value == 2)
        converted = 2; // channels last
    else if (value == 3)
        converted = 3; // channels last 3d
    else if (value == 4)
        converted = 1; // preserve
    else
        return -1;

    return 0;
}

static int exported_scalar_type_to_pnnx(int64_t value, int& converted)
{
    if (value >= 1 && value <= 12)
        converted = (int)value - 1;
    else if (value == 13)
        converted = 15; // bfloat16
    else
        return -1;

    return 0;
}

static int exported_layout_to_pnnx(int64_t value, int& converted)
{
    if (value != 7) // strided
        return -1;

    converted = 0;
    return 0;
}

static int exported_argument_to_parameter(const ExportedArgument& argument, Parameter& parameter)
{
    if (argument.type == EXPORTED_ARGUMENT_NONE)
    {
        parameter.type = 0;
    }
    else if (argument.type == EXPORTED_ARGUMENT_INT)
    {
        parameter.type = 2;
        parameter.i = exported_int_to_pnnx(argument.int_value);
    }
    else if (argument.type == EXPORTED_ARGUMENT_INT_LIST)
    {
        parameter.type = 5;
        parameter.ai.reserve(argument.int_values.size());
        for (size_t i = 0; i < argument.int_values.size(); i++)
            parameter.ai.push_back(exported_int_to_pnnx(argument.int_values[i]));
    }
    else if (argument.type == EXPORTED_ARGUMENT_FLOAT)
    {
        parameter.type = 3;
        parameter.f = (float)argument.float_value;
    }
    else if (argument.type == EXPORTED_ARGUMENT_FLOAT_LIST)
    {
        parameter.type = 6;
        parameter.af.reserve(argument.float_values.size());
        for (size_t i = 0; i < argument.float_values.size(); i++)
            parameter.af.push_back((float)argument.float_values[i]);
    }
    else if (argument.type == EXPORTED_ARGUMENT_COMPLEX)
    {
        parameter.type = 10;
        parameter.c = std::complex<float>((float)argument.complex_real_value, (float)argument.complex_imag_value);
    }
    else if (argument.type == EXPORTED_ARGUMENT_BOOL)
    {
        parameter.type = 1;
        parameter.b = argument.bool_value;
    }
    else if (argument.type == EXPORTED_ARGUMENT_STRING)
    {
        parameter.type = 4;
        parameter.s = argument.string_value;
    }
    else if (argument.type == EXPORTED_ARGUMENT_MEMORY_FORMAT)
    {
        parameter.type = 2;
        if (exported_memory_format_to_pnnx(argument.enum_value, parameter.i) != 0)
            return -1;
    }
    else if (argument.type == EXPORTED_ARGUMENT_SCALAR_TYPE)
    {
        parameter.type = 2;
        if (exported_scalar_type_to_pnnx(argument.enum_value, parameter.i) != 0)
            return -1;
    }
    else if (argument.type == EXPORTED_ARGUMENT_DEVICE)
    {
        if (argument.device_value.type.empty())
            return -1;
        if (argument.device_value.has_index && (argument.device_value.index < 0 || argument.device_value.index > 127))
            return -1;

        parameter.type = 4;
        parameter.s = argument.device_value.type;
        if (argument.device_value.has_index)
        {
            std::ostringstream device;
            device << argument.device_value.type << ':' << argument.device_value.index;
            parameter.s = device.str();
        }
    }
    else if (argument.type == EXPORTED_ARGUMENT_LAYOUT)
    {
        parameter.type = 2;
        if (exported_layout_to_pnnx(argument.enum_value, parameter.i) != 0)
            return -1;
    }
    else
        return -1;

    return 0;
}

int lower_exported_program(const ExportedProgram& source_program,
                           const std::map<std::string, MaterializedExportedTensor>& state,
                           Graph& graph,
                           std::string& error)
{
    error.clear();
    if (!graph.ops.empty() || !graph.operands.empty())
    {
        error = "destination graph must be empty";
        return -1;
    }

    ExportedProgram program = source_program;
    if (normalize_exported_program_graph(source_program.graph, program.graph, error) != 0)
        return -1;

    if (program.input_specs.size() != program.graph.inputs.size())
    {
        error = "input spec count does not match graph inputs";
        return -1;
    }
    if (program.output_specs.size() != program.graph.outputs.size())
    {
        error = "output spec count does not match graph outputs";
        return -1;
    }
    if (validate_signature_kinds(program, error) != 0)
        return -1;
    if (validate_signature_arguments(program, error) != 0)
        return -1;
    if (program.output_tree_spec.type != EXPORTED_TREE_SPEC_NONE)
    {
        size_t leaf_count = 0;
        if (validate_output_tree(program.output_tree_spec, leaf_count, error) != 0)
            return -1;
        if (leaf_count != program.graph.outputs.size())
        {
            error = "output treespec leaf count does not match graph outputs";
            return -1;
        }
    }

    Graph candidate;
    std::map<std::string, Operand*> values;
    std::set<std::string> operand_names;
    std::set<std::string> operator_names;
    int input_index = 0;
    int unknown_index = 0;

    for (size_t i = 0; i < program.input_specs.size(); i++)
    {
        const ExportedInputSpec& spec = program.input_specs[i];
        const std::string& name = spec.arg.name;
        if (values.find(name) != values.end())
        {
            error = "tensor value " + name + " is defined more than once";
            return -1;
        }

        if (spec.kind == EXPORTED_PARAMETER || spec.kind == EXPORTED_BUFFER || spec.kind == EXPORTED_TENSOR_CONSTANT)
        {
            const std::map<std::string, MaterializedExportedTensor>::const_iterator state_it = state.find(spec.target);
            if (state_it == state.end())
            {
                const char* state_kind = exported_state_kind_name(spec.kind);
                error = std::string(state_kind) + " " + spec.target + " is missing materialized state";
                return -1;
            }

            Operator* op = candidate.new_operator("pnnx.Attribute", unique_name(spec.target, operator_names));
            Operand* operand = candidate.new_operand(name);
            operand_names.insert(name);
            operand->producer = op;
            op->outputs.push_back(operand);
            if (set_tensor_metadata(program, name, operand, error) != 0)
                return -1;
            if (state_it->second.pnnx_type != operand->type)
            {
                error = std::string(exported_state_kind_name(spec.kind)) + " " + spec.target + " type does not match tensor metadata";
                return -1;
            }
            if (state_it->second.shape != operand->shape)
            {
                error = std::string(exported_state_kind_name(spec.kind)) + " " + spec.target + " shape does not match tensor metadata";
                return -1;
            }

            Attribute attribute;
            attribute.type = state_it->second.pnnx_type;
            attribute.shape = state_it->second.shape;
            attribute.data = state_it->second.data;
            op->attrs["data"] = attribute;
            values[name] = operand;
            continue;
        }

        std::ostringstream input_name;
        input_name << "pnnx_input_" << input_index++;
        Operator* op = candidate.new_operator("pnnx.Input", unique_name(input_name.str(), operator_names));
        Operand* operand = candidate.new_operand(name);
        operand_names.insert(name);
        operand->producer = op;
        op->outputs.push_back(operand);
        if (set_tensor_metadata(program, name, operand, error) != 0)
            return -1;
        values[name] = operand;
    }

    for (size_t i = 0; i < program.graph.nodes.size(); i++)
    {
        const ExportedNode& node = program.graph.nodes[i];
        std::vector<CanonicalExportedArgument> arguments;
        if (canonicalize_exported_arguments(node, program.header, arguments, error) != 0)
            return -1;

        ExportedAtenTarget target;
        if (parse_exported_aten_target(node.target, target, error) != 0)
            return -1;

        std::ostringstream generated_name;
        if (!node.has_name)
            generated_name << "pnnx_" << unknown_index++;
        const std::string requested_name = node.has_name ? node.name : generated_name.str();
        Operator* op = candidate.new_operator(target.operator_name, unique_name(requested_name, operator_names));

        for (size_t j = 0; j < arguments.size(); j++)
        {
            Operand* operand = 0;
            if (arguments[j].value.type == EXPORTED_ARGUMENT_TENSOR)
            {
                const std::map<std::string, Operand*>::const_iterator value_it = values.find(arguments[j].value.name);
                if (value_it == values.end())
                {
                    error = "unknown tensor value " + arguments[j].value.name + " for " + node.target;
                    return -1;
                }

                operand = value_it->second;
            }
            else if (arguments[j].value.type == EXPORTED_ARGUMENT_TENSOR_LIST)
            {
                std::ostringstream list_name;
                list_name << "pnnx_" << unknown_index++;
                Operator* list = candidate.new_operator_before("prim::ListConstruct", unique_name(list_name.str(), operator_names), op);
                operand = candidate.new_operand(unique_name(list_name.str(), operand_names));
                operand->producer = list;
                list->outputs.push_back(operand);

                for (size_t k = 0; k < arguments[j].value.tensor_names.size(); k++)
                {
                    const std::string& tensor_name = arguments[j].value.tensor_names[k];
                    const std::map<std::string, Operand*>::const_iterator value_it = values.find(tensor_name);
                    if (value_it == values.end())
                    {
                        error = "unknown tensor value " + tensor_name + " for tensor-list argument " + arguments[j].name + " of " + node.target;
                        return -1;
                    }

                    value_it->second->consumers.push_back(list);
                    list->inputs.push_back(value_it->second);
                }
            }
            else
            {
                std::ostringstream constant_name;
                constant_name << "pnnx_" << unknown_index++;
                const std::string constant_operator_name = unique_name(constant_name.str(), operator_names);
                Operator* constant = candidate.new_operator_before("prim::Constant", constant_operator_name, op);
                const std::string constant_operand_name = unique_name(constant_name.str(), operand_names);
                operand = candidate.new_operand(constant_operand_name);
                operand->producer = constant;
                constant->outputs.push_back(operand);
                if (exported_argument_to_parameter(arguments[j].value, constant->params["value"]) != 0)
                {
                    error = "unsupported non-tensor argument " + arguments[j].name + " for " + node.target;
                    return -1;
                }
            }

            operand->consumers.push_back(op);
            op->inputs.push_back(operand);
            op->inputnames.push_back(arguments[j].name);
        }

        for (size_t j = 0; j < node.outputs.size(); j++)
        {
            if (node.outputs[j].type == EXPORTED_ARGUMENT_TENSOR)
            {
                const std::string& name = node.outputs[j].name;
                if (values.find(name) != values.end())
                {
                    error = "tensor value " + name + " is defined more than once";
                    return -1;
                }
                Operand* operand = candidate.new_operand(name);
                operand_names.insert(name);
                operand->producer = op;
                op->outputs.push_back(operand);
                if (set_tensor_metadata(program, name, operand, error) != 0)
                    return -1;
                values[name] = operand;
                continue;
            }

            if (node.outputs[j].type == EXPORTED_ARGUMENT_TENSOR_LIST)
            {
                std::ostringstream list_name;
                list_name << "pnnx_" << unknown_index++;
                Operator* unpack = candidate.new_operator_after("prim::ListUnpack", unique_name(list_name.str(), operator_names), op);
                Operand* list_operand = candidate.new_operand(unique_name(list_name.str(), operand_names));
                list_operand->producer = op;
                list_operand->consumers.push_back(unpack);
                op->outputs.push_back(list_operand);
                unpack->inputs.push_back(list_operand);

                for (size_t k = 0; k < node.outputs[j].tensor_names.size(); k++)
                {
                    const std::string& name = node.outputs[j].tensor_names[k];
                    if (values.find(name) != values.end())
                    {
                        error = "tensor value " + name + " is defined more than once";
                        return -1;
                    }
                    Operand* operand = candidate.new_operand(name);
                    operand_names.insert(name);
                    operand->producer = unpack;
                    unpack->outputs.push_back(operand);
                    if (set_tensor_metadata(program, name, operand, error) != 0)
                        return -1;
                    values[name] = operand;
                }
                continue;
            }

            if (node.outputs[j].type == EXPORTED_ARGUMENT_NONE)
                continue;

            error = "unsupported non-tensor output for " + node.target;
            return -1;
        }
    }

    std::vector<Operand*> flat_outputs;
    flat_outputs.reserve(program.output_specs.size());
    for (size_t i = 0; i < program.output_specs.size(); i++)
    {
        const ExportedOutputSpec& spec = program.output_specs[i];
        const std::map<std::string, Operand*>::const_iterator value_it = values.find(spec.arg.name);
        if (value_it == values.end())
        {
            error = "unknown graph output " + spec.arg.name;
            return -1;
        }

        flat_outputs.push_back(value_it->second);
    }

    if (program.output_tree_spec.type != EXPORTED_TREE_SPEC_NONE)
    {
        size_t flat_index = 0;
        Operand* tree_output = 0;
        if (construct_output_tree(program.output_tree_spec, flat_outputs, flat_index, candidate, operand_names, operator_names, unknown_index, tree_output, error) != 0)
            return -1;
        if (flat_index != flat_outputs.size())
        {
            error = "output treespec did not consume all graph outputs";
            return -1;
        }

        Operator* op = candidate.new_operator("pnnx.Output", unique_name("pnnx_output_0", operator_names));
        tree_output->consumers.push_back(op);
        op->inputs.push_back(tree_output);
    }
    else
    {
        for (size_t i = 0; i < flat_outputs.size(); i++)
        {
            std::ostringstream output_name;
            output_name << "pnnx_output_" << i;
            Operator* op = candidate.new_operator("pnnx.Output", unique_name(output_name.str(), operator_names));
            Operand* operand = flat_outputs[i];
            operand->consumers.push_back(op);
            op->inputs.push_back(operand);
        }
    }

    graph.ops.swap(candidate.ops);
    graph.operands.swap(candidate.operands);
    return 0;
}

static int parse_program_json(const JsonValue& value, const std::string& entry, ExportedProgram& program, std::string& error)
{
    ExportedSchemaError schema_error;
    if (parse_exported_program(value, program, schema_error) != 0)
    {
        error = "invalid exported program " + entry + " at " + schema_error.path + ": " + schema_error.message;
        return -1;
    }

    return 0;
}

static int parse_payload_json(const JsonValue& value, const std::string& entry, ExportedPayloadConfig& config, std::string& error)
{
    ExportedSchemaError schema_error;
    if (parse_exported_payload_config(value, config, schema_error) != 0)
    {
        error = "invalid exported payload config " + entry + " at " + schema_error.path + ": " + schema_error.message;
        return -1;
    }

    return 0;
}

static int materialize_program_state(Pt2ArchiveReader& reader,
                                     const ExportedProgram& program,
                                     const ExportedPayloadConfig& weights,
                                     const ExportedPayloadConfig& constants,
                                     std::map<std::string, MaterializedExportedTensor>& state,
                                     std::string& error)
{
    std::map<std::string, std::vector<char> > storage_cache;

    for (size_t i = 0; i < program.input_specs.size(); i++)
    {
        const ExportedInputSpec& spec = program.input_specs[i];
        if (spec.kind != EXPORTED_PARAMETER && spec.kind != EXPORTED_BUFFER && spec.kind != EXPORTED_TENSOR_CONSTANT)
            continue;

        const bool is_constant = spec.kind == EXPORTED_TENSOR_CONSTANT || (spec.kind == EXPORTED_BUFFER && !spec.persistent);
        const ExportedPayloadConfig& config = is_constant ? constants : weights;
        const ExportedPayloadConfig& wrong_config = is_constant ? weights : constants;
        const char* config_name = is_constant ? "constants" : "weights";
        const char* wrong_config_name = is_constant ? "weights" : "constants";
        const char* state_kind = exported_state_kind_name(spec.kind);

        if (wrong_config.entries.find(spec.target) != wrong_config.entries.end())
        {
            error = std::string(state_kind) + " " + spec.target + " is present in " + wrong_config_name + " config";
            return -1;
        }

        const std::map<std::string, ExportedPayloadEntry>::const_iterator entry_it = config.entries.find(spec.target);
        if (entry_it == config.entries.end())
        {
            error = std::string(state_kind) + " " + spec.target + " is missing from " + config_name + " config";
            return -1;
        }

        const ExportedPayloadEntry& entry = entry_it->second;
        const std::string directory = is_constant ? "/data/constants/" : "/data/weights/";
        const std::string payload_path = reader.layout().root + directory + entry.path_name;

        const bool expected_is_param = spec.kind == EXPORTED_PARAMETER;
        if (entry.is_param != expected_is_param)
        {
            error = std::string(state_kind) + " " + spec.target + " in " + config_name + " config has is_param=" + (entry.is_param ? "true" : "false");
            return -1;
        }

        if (entry.use_pickle || !entry.has_tensor_meta)
        {
            error = std::string("pickled payload is unsupported for ") + state_kind + " " + spec.target + " at " + payload_path;
            return -1;
        }

        std::map<std::string, std::vector<char> >::iterator storage_it = storage_cache.find(payload_path);
        if (storage_it == storage_cache.end())
        {
            storage_it = storage_cache.insert(std::make_pair(payload_path, std::vector<char>())).first;
            if (reader.read_blob(payload_path, storage_it->second, error) != 0)
                return -1;
        }

        MaterializedExportedTensor tensor;
        if (materialize_exported_tensor(entry.tensor_meta, storage_it->second, reader.layout().byte_order, tensor, error) != 0)
        {
            error = spec.target + " from " + payload_path + ": " + error;
            return -1;
        }

        state[spec.target] = std::move(tensor);
    }

    return 0;
}

int load_exported_program(const std::string& pt2path, Graph& graph, std::string& error)
{
    error.clear();

    Pt2ArchiveReader reader;
    if (reader.open(pt2path, error) != 0)
        return -1;

    ExportedProgram program;
    ExportedPayloadConfig weights;
    ExportedPayloadConfig constants;
    {
        JsonValue model_json;
        if (reader.read_json(reader.layout().model_json_path, model_json, error) != 0)
            return -1;
        if (parse_program_json(model_json, reader.layout().model_json_path, program, error) != 0)
            return -1;
    }
    {
        JsonValue weights_json;
        if (reader.read_json(reader.layout().weights_config_path, weights_json, error) != 0)
            return -1;
        if (parse_payload_json(weights_json, reader.layout().weights_config_path, weights, error) != 0)
            return -1;
    }
    {
        JsonValue constants_json;
        if (reader.read_json(reader.layout().constants_config_path, constants_json, error) != 0)
            return -1;
        if (parse_payload_json(constants_json, reader.layout().constants_config_path, constants, error) != 0)
            return -1;
    }

    std::map<std::string, MaterializedExportedTensor> state;
    if (materialize_program_state(reader, program, weights, constants, state, error) != 0)
        return -1;

    return lower_exported_program(program, state, graph, error);
}

} // namespace pnnx
