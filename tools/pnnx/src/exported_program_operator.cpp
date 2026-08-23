// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "exported_program_operator.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/ivalue.h>
#include <c10/core/DeviceType.h>
#include <torch/csrc/api/include/torch/version.h>
#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 9)
#include <torch/csrc/jit/operator_upgraders/utils.h>
#endif

#include <exception>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace pnnx {

static bool is_identifier(const std::string& value)
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

int parse_exported_aten_target(const std::string& target, ExportedAtenTarget& result, std::string& error)
{
    result = ExportedAtenTarget();
    error.clear();

    const std::string prefix = "torch.ops.aten.";
    if (target.compare(0, prefix.size(), prefix) != 0)
    {
        error = "exported operator target must start with torch.ops.aten.";
        return -1;
    }

    const std::string qualified_name = target.substr(prefix.size());
    const size_t separator = qualified_name.rfind('.');
    if (separator == std::string::npos)
    {
        error = "exported operator target must contain operator and overload";
        return -1;
    }

    const std::string operator_part = qualified_name.substr(0, separator);
    const std::string overload_part = qualified_name.substr(separator + 1);
    if (!is_identifier(operator_part))
    {
        error = "invalid operator name in exported target";
        return -1;
    }
    if (!is_identifier(overload_part))
    {
        error = "invalid overload name in exported target";
        return -1;
    }

    ExportedAtenTarget parsed_target;
    parsed_target.operator_name = "aten::" + operator_part;
    if (overload_part != "default")
        parsed_target.overload_name = overload_part;

    result = parsed_target;
    return 0;
}

bool is_exported_aten_target_supported(const ExportedAtenTarget& target)
{
    if (target.operator_name == "aten::add_" && target.overload_name == "Tensor")
        return true;
    if (target.operator_name == "aten::flatten" && target.overload_name == "using_ints")
        return true;

    if (!target.overload_name.empty())
        return false;

    return target.operator_name == "aten::adaptive_avg_pool2d" || target.operator_name == "aten::batch_norm" || target.operator_name == "aten::conv2d" || target.operator_name == "aten::linear" || target.operator_name == "aten::max_pool2d" || target.operator_name == "aten::relu" || target.operator_name == "aten::relu_";
}

static std::string operator_context(const ExportedProgramHeader& header, const ExportedNode& node)
{
    std::ostringstream context;
    context << "torch " << (header.torch_version.empty() ? "unknown" : header.torch_version) << " aten opset ";

    const std::map<std::string, int64_t>::const_iterator aten_opset = header.opset_version.find("aten");
    if (aten_opset == header.opset_version.end())
        context << "missing";
    else
        context << aten_opset->second;

    context << " target " << node.target << ": ";
    return context.str();
}

static int operator_error(const ExportedProgramHeader& header, const ExportedNode& node, const std::string& message, std::string& error)
{
    error = operator_context(header, node) + message;
    return -1;
}

#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 9)

static c10::TypePtr unwrap_optional_type(const c10::TypePtr& type)
{
    const c10::OptionalTypePtr optional_type = type->cast<c10::OptionalType>();
    if (optional_type)
        return optional_type->getElementType();
    return type;
}

static int serialize_scalar_type(at::ScalarType value, int64_t& serialized)
{
    if (value == at::ScalarType::Byte)
        serialized = 1;
    else if (value == at::ScalarType::Char)
        serialized = 2;
    else if (value == at::ScalarType::Short)
        serialized = 3;
    else if (value == at::ScalarType::Int)
        serialized = 4;
    else if (value == at::ScalarType::Long)
        serialized = 5;
    else if (value == at::ScalarType::Half)
        serialized = 6;
    else if (value == at::ScalarType::Float)
        serialized = 7;
    else if (value == at::ScalarType::Double)
        serialized = 8;
    else if (value == at::ScalarType::ComplexHalf)
        serialized = 9;
    else if (value == at::ScalarType::ComplexFloat)
        serialized = 10;
    else if (value == at::ScalarType::ComplexDouble)
        serialized = 11;
    else if (value == at::ScalarType::Bool)
        serialized = 12;
    else if (value == at::ScalarType::BFloat16)
        serialized = 13;
    else
        return -1;

    return 0;
}

static int serialize_layout(at::Layout value, int64_t& serialized)
{
    if (value == at::kSparse)
        serialized = 1;
    else if (value == at::kSparseCsr)
        serialized = 2;
    else if (value == at::kSparseCsc)
        serialized = 3;
    else if (value == at::kSparseBsr)
        serialized = 4;
    else if (value == at::kSparseBsc)
        serialized = 5;
    else if (value == at::kMkldnn)
        serialized = 6;
    else if (value == at::kStrided)
        serialized = 7;
    else
        return -1;

    return 0;
}

static int serialize_memory_format(at::MemoryFormat value, int64_t& serialized)
{
    if (value == at::MemoryFormat::Contiguous)
        serialized = 1;
    else if (value == at::MemoryFormat::ChannelsLast)
        serialized = 2;
    else if (value == at::MemoryFormat::ChannelsLast3d)
        serialized = 3;
    else if (value == at::MemoryFormat::Preserve)
        serialized = 4;
    else
        return -1;

    return 0;
}

static int convert_default_value(const c10::Argument& schema_argument, ExportedArgument& argument, std::string& error)
{
    argument = ExportedArgument();
    error.clear();

    if (!schema_argument.default_value())
    {
        error = "argument has no default value";
        return -1;
    }

    const c10::IValue& value = *schema_argument.default_value();
    const c10::TypePtr argument_type = unwrap_optional_type(schema_argument.real_type());

    if (value.isNone())
    {
        argument.type = EXPORTED_ARGUMENT_NONE;
        return 0;
    }
    if (value.isBool())
    {
        argument.type = EXPORTED_ARGUMENT_BOOL;
        argument.bool_value = value.toBool();
        return 0;
    }
    if (value.isInt())
    {
        if (argument_type->kind() == c10::TypeKind::ScalarTypeType)
        {
            argument.type = EXPORTED_ARGUMENT_SCALAR_TYPE;
            if (serialize_scalar_type(value.toScalarType(), argument.enum_value) != 0)
            {
                error = "unsupported ScalarType default";
                return -1;
            }
            return 0;
        }
        if (argument_type->kind() == c10::TypeKind::LayoutType)
        {
            argument.type = EXPORTED_ARGUMENT_LAYOUT;
            if (serialize_layout(value.toLayout(), argument.enum_value) != 0)
            {
                error = "unsupported Layout default";
                return -1;
            }
            return 0;
        }
        if (argument_type->kind() == c10::TypeKind::MemoryFormatType)
        {
            argument.type = EXPORTED_ARGUMENT_MEMORY_FORMAT;
            if (serialize_memory_format(value.toMemoryFormat(), argument.enum_value) != 0)
            {
                error = "unsupported MemoryFormat default";
                return -1;
            }
            return 0;
        }

        argument.type = EXPORTED_ARGUMENT_INT;
        argument.int_value = value.toInt();
        return 0;
    }
    if (value.isDouble())
    {
        argument.type = EXPORTED_ARGUMENT_FLOAT;
        argument.float_value = value.toDouble();
        return 0;
    }
    if (value.isString())
    {
        argument.type = EXPORTED_ARGUMENT_STRING;
        argument.string_value = value.toStringRef();
        return 0;
    }
    if (value.isIntList())
    {
        argument.type = EXPORTED_ARGUMENT_INT_LIST;
        argument.int_values = value.toIntVector();
        return 0;
    }
    if (value.isDoubleList())
    {
        argument.type = EXPORTED_ARGUMENT_FLOAT_LIST;
        argument.float_values = value.toDoubleVector();
        return 0;
    }
    if (value.isBoolList())
    {
        argument.type = EXPORTED_ARGUMENT_BOOL_LIST;
        const c10::List<bool> values = value.toBoolList();
        argument.bool_values.reserve(values.size());
        for (size_t i = 0; i < values.size(); i++)
            argument.bool_values.push_back(values.get(i));
        return 0;
    }
    if (value.isTensorList())
    {
        const c10::List<at::Tensor> values = value.toTensorList();
        if (!values.empty())
        {
            error = "non-empty tensor-list default cannot be represented";
            return -1;
        }
        argument.type = EXPORTED_ARGUMENT_TENSOR_LIST;
        return 0;
    }
    if (value.isList())
    {
        const c10::List<c10::IValue> values = value.toList();
        if (values.elementType()->kind() != c10::TypeKind::StringType)
        {
            error = "generic list default is not a string list";
            return -1;
        }

        bool all_strings = true;
        for (size_t i = 0; i < values.size(); i++)
            all_strings = all_strings && values.get(i).isString();

        if (!all_strings)
        {
            error = "generic list default is not a string list";
            return -1;
        }

        argument.type = EXPORTED_ARGUMENT_STRING_LIST;
        argument.string_values.reserve(values.size());
        for (size_t i = 0; i < values.size(); i++)
            argument.string_values.push_back(values.get(i).toStringRef());
        return 0;
    }
    if (value.isDevice())
    {
        const c10::Device device = value.toDevice();
        argument.type = EXPORTED_ARGUMENT_DEVICE;
        argument.device_value.type = c10::DeviceTypeName(device.type(), true);
        argument.device_value.has_index = device.has_index();
        if (device.has_index())
            argument.device_value.index = static_cast<int64_t>(static_cast<unsigned char>(device.index()));
        return 0;
    }

    error = "unsupported default IValue kind " + value.tagKind();
    return -1;
}

static int canonicalize_with_schema(const ExportedNode& node,
                                    const ExportedProgramHeader& header,
                                    const c10::FunctionSchema& schema,
                                    std::vector<CanonicalExportedArgument>& result,
                                    std::string& error)
{
    if (schema.is_vararg())
        return operator_error(header, node, "variadic dispatcher schemas are unsupported", error);

    const std::vector<c10::Argument>& schema_arguments = schema.arguments();
    std::map<std::string, size_t> schema_indices;
    for (size_t i = 0; i < schema_arguments.size(); i++)
        schema_indices[schema_arguments[i].name()] = i;

    bool has_missing_kind = false;
    bool has_explicit_kind = false;
    for (size_t i = 0; i < node.inputs.size(); i++)
    {
        if (node.inputs[i].kind == EXPORTED_ARGUMENT_KIND_UNKNOWN)
            return operator_error(header, node, "unknown argument kind for " + node.inputs[i].name, error);
        if (node.inputs[i].kind == EXPORTED_ARGUMENT_KIND_MISSING)
            has_missing_kind = true;
        else
            has_explicit_kind = true;
    }
    if (has_missing_kind && has_explicit_kind)
        return operator_error(header, node, "node mixes legacy and explicit argument kinds", error);

    std::vector<const ExportedNamedArgument*> bound_arguments(schema_arguments.size(), 0);
    bool saw_keyword = false;
    size_t next_positional = 0;
    for (size_t i = 0; i < node.inputs.size(); i++)
    {
        const ExportedNamedArgument& input = node.inputs[i];
        const std::map<std::string, size_t>::const_iterator schema_index_it = schema_indices.find(input.name);
        if (schema_index_it == schema_indices.end())
            return operator_error(header, node, "unknown argument " + input.name, error);

        const size_t schema_index = schema_index_it->second;
        if (bound_arguments[schema_index])
            return operator_error(header, node, "duplicate argument " + input.name, error);

        if (!has_missing_kind && input.kind == EXPORTED_ARGUMENT_KIND_POSITIONAL)
        {
            if (saw_keyword)
                return operator_error(header, node, "positional argument " + input.name + " follows a keyword argument", error);
            if (schema_arguments[schema_index].kwarg_only())
                return operator_error(header, node, "keyword-only argument " + input.name + " was serialized as positional", error);
            if (schema_index != next_positional)
            {
                std::string expected_name = next_positional < schema_arguments.size() ? schema_arguments[next_positional].name() : std::string("<none>");
                return operator_error(header, node, "expected positional argument " + expected_name + " but found " + input.name, error);
            }
            next_positional++;
        }
        else if (!has_missing_kind && input.kind == EXPORTED_ARGUMENT_KIND_KEYWORD)
        {
            saw_keyword = true;
        }

        if (input.arg.type == EXPORTED_ARGUMENT_UNSUPPORTED)
            return operator_error(header, node, "unsupported serialized argument " + input.arg.unsupported_tag + " for " + input.name, error);

        bound_arguments[schema_index] = &input;
    }

    std::vector<CanonicalExportedArgument> canonical_arguments;
    canonical_arguments.reserve(schema_arguments.size());
    for (size_t i = 0; i < schema_arguments.size(); i++)
    {
        CanonicalExportedArgument argument;
        argument.name = schema_arguments[i].name();
        if (bound_arguments[i])
        {
            argument.value = bound_arguments[i]->arg;
        }
        else
        {
            if (!schema_arguments[i].default_value())
                return operator_error(header, node, "missing required argument " + argument.name, error);

            std::string default_error;
            if (convert_default_value(schema_arguments[i], argument.value, default_error) != 0)
                return operator_error(header, node, "cannot materialize default for argument " + argument.name + ": " + default_error, error);
        }
        canonical_arguments.push_back(argument);
    }

    result.swap(canonical_arguments);
    return 0;
}

#endif

int canonicalize_exported_arguments(const ExportedNode& node,
                                    const ExportedProgramHeader& header,
                                    std::vector<CanonicalExportedArgument>& result,
                                    std::string& error)
{
    result.clear();
    error.clear();

    ExportedAtenTarget target;
    std::string target_error;
    if (parse_exported_aten_target(node.target, target, target_error) != 0)
        return operator_error(header, node, target_error, error);

    const std::map<std::string, int64_t>::const_iterator aten_opset = header.opset_version.find("aten");
    if (aten_opset == header.opset_version.end())
        return operator_error(header, node, "missing aten opset", error);

#if TORCH_VERSION_MAJOR < 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR < 9)
    return operator_error(header, node, "exported program operator schemas require libtorch 2.9 or newer", error);
#else
    const uint64_t linked_opset = torch::jit::getMaxOperatorVersion();
    if (aten_opset->second < 0 || (uint64_t)aten_opset->second != linked_opset)
    {
        std::ostringstream message;
        message << "archive aten opset " << aten_opset->second << " does not match linked libtorch opset " << linked_opset;
        return operator_error(header, node, message.str(), error);
    }

    try
    {
        const c10::OperatorHandle operator_handle = c10::Dispatcher::singleton().findSchemaOrThrow(target.operator_name.c_str(), target.overload_name.c_str());
        const c10::FunctionSchema& schema = operator_handle.schema();
        return canonicalize_with_schema(node, header, schema, result, error);
    }
    catch (const c10::Error& e)
    {
        return operator_error(header, node, "cannot resolve dispatcher schema: " + std::string(e.what_without_backtrace()), error);
    }
    catch (const std::exception& e)
    {
        return operator_error(header, node, "dispatcher schema failure: " + std::string(e.what()), error);
    }
#endif
}

} // namespace pnnx
