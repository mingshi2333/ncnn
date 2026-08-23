// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "load_exported_program.h"

#include "exported_program_operator.h"
#include "pt2_archive.h"

#include <limits.h>

#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace pnnx {

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
        if (meta.sizes[i] < 0)
        {
            std::ostringstream message;
            message << "tensor " << name << " has a negative or symbolic size at dimension " << i;
            error = message.str();
            return -1;
        }
        if (meta.sizes[i] > INT_MAX)
        {
            error = "tensor shape does not fit pnnx for " + name;
            return -1;
        }
        shape.push_back((int)meta.sizes[i]);
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
            if (!spec.persistent)
            {
                error = "non-persistent buffer " + spec.target + " is unsupported";
                return -1;
            }
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

static int exported_int_to_pnnx(int64_t value)
{
    if (value == std::numeric_limits<int64_t>::max()) value = INT_MAX;
    if (value == std::numeric_limits<int64_t>::max() - 1) value = INT_MAX - 1;
    if (value == std::numeric_limits<int64_t>::min()) value = INT_MIN;
    if (value == std::numeric_limits<int64_t>::min() + 1) value = INT_MIN + 1;
    return (int)value;
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
    else if (argument.type == EXPORTED_ARGUMENT_BOOL)
    {
        parameter.type = 1;
        parameter.b = argument.bool_value;
    }
    else
        return -1;

    return 0;
}

int lower_exported_program(const ExportedProgram& program,
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
            if (node.outputs[j].type != EXPORTED_ARGUMENT_TENSOR)
            {
                error = "unsupported non-tensor output for " + node.target;
                return -1;
            }

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
        }
    }

    for (size_t i = 0; i < program.output_specs.size(); i++)
    {
        const ExportedOutputSpec& spec = program.output_specs[i];
        const std::map<std::string, Operand*>::const_iterator value_it = values.find(spec.arg.name);
        if (value_it == values.end())
        {
            error = "unknown graph output " + spec.arg.name;
            return -1;
        }

        std::ostringstream output_name;
        output_name << "pnnx_output_" << i;
        Operator* op = candidate.new_operator("pnnx.Output", unique_name(output_name.str(), operator_names));
        Operand* operand = value_it->second;
        operand->consumers.push_back(op);
        op->inputs.push_back(operand);
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

        const bool is_constant = spec.kind == EXPORTED_TENSOR_CONSTANT;
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
