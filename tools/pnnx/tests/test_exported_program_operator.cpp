// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "exported_program_operator.h"

#include "pt2_archive.h"

#include <stdio.h>

#include <map>
#include <sstream>
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

static pnnx::ExportedProgramHeader make_header(int64_t aten_opset = 10)
{
    pnnx::ExportedProgramHeader header;
    header.schema_major = 8;
    header.schema_minor = 20;
    header.torch_version = "2.12.1+cu126";
    if (aten_opset >= 0)
        header.opset_version["aten"] = aten_opset;
    return header;
}

static pnnx::ExportedArgument make_none()
{
    return pnnx::ExportedArgument();
}

static pnnx::ExportedArgument make_tensor(const std::string& name)
{
    pnnx::ExportedArgument value;
    value.type = pnnx::EXPORTED_ARGUMENT_TENSOR;
    value.name = name;
    return value;
}

static pnnx::ExportedArgument make_int(int64_t number)
{
    pnnx::ExportedArgument value;
    value.type = pnnx::EXPORTED_ARGUMENT_INT;
    value.int_value = number;
    return value;
}

static pnnx::ExportedArgument make_ints(const std::vector<int64_t>& numbers)
{
    pnnx::ExportedArgument value;
    value.type = pnnx::EXPORTED_ARGUMENT_INT_LIST;
    value.int_values = numbers;
    return value;
}

static pnnx::ExportedArgument make_float(double number)
{
    pnnx::ExportedArgument value;
    value.type = pnnx::EXPORTED_ARGUMENT_FLOAT;
    value.float_value = number;
    return value;
}

static pnnx::ExportedArgument make_complex(double real, double imag)
{
    pnnx::ExportedArgument value;
    value.type = pnnx::EXPORTED_ARGUMENT_COMPLEX;
    value.complex_real_value = real;
    value.complex_imag_value = imag;
    return value;
}

static pnnx::ExportedArgument make_bool(bool boolean)
{
    pnnx::ExportedArgument value;
    value.type = pnnx::EXPORTED_ARGUMENT_BOOL;
    value.bool_value = boolean;
    return value;
}

static pnnx::ExportedArgument make_string(const std::string& text)
{
    pnnx::ExportedArgument value;
    value.type = pnnx::EXPORTED_ARGUMENT_STRING;
    value.string_value = text;
    return value;
}

static pnnx::ExportedArgument make_enum(pnnx::ExportedArgumentType type, int64_t number)
{
    pnnx::ExportedArgument value;
    value.type = type;
    value.enum_value = number;
    return value;
}

static pnnx::ExportedArgument make_unsupported(const std::string& tag)
{
    pnnx::ExportedArgument value;
    value.type = pnnx::EXPORTED_ARGUMENT_UNSUPPORTED;
    value.unsupported_tag = tag;
    return value;
}

static pnnx::ExportedNamedArgument make_input(const std::string& name, const pnnx::ExportedArgument& value, pnnx::ExportedArgumentKind kind)
{
    pnnx::ExportedNamedArgument input;
    input.name = name;
    input.arg = value;
    input.kind = kind;
    return input;
}

static pnnx::CanonicalExportedArgument make_expected(const std::string& name, const pnnx::ExportedArgument& value)
{
    pnnx::CanonicalExportedArgument argument;
    argument.name = name;
    argument.value = value;
    return argument;
}

static pnnx::ExportedNode make_node(const std::string& target, const std::vector<pnnx::ExportedNamedArgument>& inputs)
{
    pnnx::ExportedNode node;
    node.target = target;
    node.inputs = inputs;
    return node;
}

static bool same_argument(const pnnx::ExportedArgument& actual, const pnnx::ExportedArgument& expected)
{
    if (actual.type != expected.type)
        return false;

    if (actual.type == pnnx::EXPORTED_ARGUMENT_NONE)
        return true;
    if (actual.type == pnnx::EXPORTED_ARGUMENT_TENSOR)
        return actual.name == expected.name;
    if (actual.type == pnnx::EXPORTED_ARGUMENT_TENSOR_LIST)
        return actual.tensor_names == expected.tensor_names;
    if (actual.type == pnnx::EXPORTED_ARGUMENT_INT)
        return actual.int_value == expected.int_value;
    if (actual.type == pnnx::EXPORTED_ARGUMENT_INT_LIST)
        return actual.int_values == expected.int_values;
    if (actual.type == pnnx::EXPORTED_ARGUMENT_FLOAT)
        return actual.float_value == expected.float_value;
    if (actual.type == pnnx::EXPORTED_ARGUMENT_FLOAT_LIST)
        return actual.float_values == expected.float_values;
    if (actual.type == pnnx::EXPORTED_ARGUMENT_COMPLEX)
        return actual.complex_real_value == expected.complex_real_value && actual.complex_imag_value == expected.complex_imag_value;
    if (actual.type == pnnx::EXPORTED_ARGUMENT_BOOL)
        return actual.bool_value == expected.bool_value;
    if (actual.type == pnnx::EXPORTED_ARGUMENT_BOOL_LIST)
        return actual.bool_values == expected.bool_values;
    if (actual.type == pnnx::EXPORTED_ARGUMENT_STRING)
        return actual.string_value == expected.string_value;
    if (actual.type == pnnx::EXPORTED_ARGUMENT_STRING_LIST)
        return actual.string_values == expected.string_values;
    if (actual.type == pnnx::EXPORTED_ARGUMENT_SCALAR_TYPE || actual.type == pnnx::EXPORTED_ARGUMENT_MEMORY_FORMAT || actual.type == pnnx::EXPORTED_ARGUMENT_LAYOUT)
        return actual.enum_value == expected.enum_value;
    if (actual.type == pnnx::EXPORTED_ARGUMENT_DEVICE)
        return actual.device_value.type == expected.device_value.type && actual.device_value.index == expected.device_value.index && actual.device_value.has_index == expected.device_value.has_index;

    return actual.unsupported_tag == expected.unsupported_tag;
}

static void expect_target(const std::string& serialized, const std::string& operator_name, const std::string& overload_name)
{
    pnnx::ExportedOperatorTarget target;
    target.operator_name = "stale";
    target.overload_name = "stale";
    std::string error = "stale";

    const int result = pnnx::parse_exported_operator_target(serialized, target, error);
    check(result == 0, "target parse", serialized + " failed: " + error);
    check(error.empty(), "target parse", serialized + " retained an error");
    const size_t namespace_separator = operator_name.find("::");
    check(target.namespace_name == operator_name.substr(0, namespace_separator), "target parse", serialized + " has wrong namespace");
    check(target.operator_name == operator_name, "target parse", serialized + " has wrong operator name");
    check(target.overload_name == overload_name, "target parse", serialized + " has wrong overload name");
}

static void expect_target_error(const std::string& serialized, const std::string& expected_error)
{
    pnnx::ExportedOperatorTarget target;
    target.operator_name = "stale";
    target.overload_name = "stale";
    std::string error = "stale";

    const int result = pnnx::parse_exported_operator_target(serialized, target, error);
    check(result != 0, "target error", serialized + " unexpectedly parsed");
    check(error.find(expected_error) != std::string::npos, "target error", serialized + " has wrong error " + error);
    check(target.namespace_name.empty() && target.operator_name.empty() && target.overload_name.empty(), "target error", serialized + " retained partial output");
}

static void expect_canonical(const pnnx::ExportedNode& node, const std::vector<pnnx::CanonicalExportedArgument>& expected, const char* name)
{
    std::vector<pnnx::CanonicalExportedArgument> actual(1, make_expected("stale", make_int(99)));
    std::string error = "stale";
    const int result = pnnx::canonicalize_exported_arguments(node, make_header(), actual, error);
    if (result != 0)
    {
        check(false, name, "unexpected error " + error);
        return;
    }

    check(error.empty(), name, "success retained an error");
    check(actual.size() == expected.size(), name, "wrong canonical argument count");
    const size_t count = actual.size() < expected.size() ? actual.size() : expected.size();
    for (size_t i = 0; i < count; i++)
    {
        std::ostringstream item;
        item << "argument " << i;
        check(actual[i].name == expected[i].name, name, item.str() + " has wrong name " + actual[i].name);
        check(same_argument(actual[i].value, expected[i].value), name, item.str() + " has wrong value");
    }
}

static void expect_canonical_error(const pnnx::ExportedNode& node, const pnnx::ExportedProgramHeader& header, const std::string& expected_error, const char* name)
{
    std::vector<pnnx::CanonicalExportedArgument> actual(1, make_expected("stale", make_int(99)));
    std::string error = "stale";
    const int result = pnnx::canonicalize_exported_arguments(node, header, actual, error);

    check(result != 0, name, "canonicalization unexpectedly succeeded");
    check(error.find(expected_error) != std::string::npos, name, "wrong error " + error);
    check(error.find(header.torch_version) != std::string::npos, name, "error lost producer version " + error);
    check(error.find(node.target) != std::string::npos, name, "error lost node target " + error);
    check(actual.empty(), name, "failure retained partial arguments");
}

static void test_targets()
{
    expect_target("torch.ops.aten.linear.default", "aten::linear", "");
    expect_target("torch.ops.aten.add.Tensor", "aten::add", "Tensor");
    expect_target("torch.ops.aten.flatten.using_ints", "aten::flatten", "using_ints");
    expect_target("torch.ops.aten._to_copy.default", "aten::_to_copy", "");
    expect_target("torch.ops.torchvision.deform_conv2d.default", "torchvision::deform_conv2d", "");
    expect_target("torch.ops.torchvision.roi_align.default", "torchvision::roi_align", "");

    expect_target_error("", "must start");
    expect_target_error("aten.add.Tensor", "must start");
    expect_target_error("torch.ops.aten.add", "operator and overload");
    expect_target_error("torch.ops..add.default", "invalid namespace");
    expect_target_error("torch.ops.aten..default", "invalid operator name");
    expect_target_error("torch.ops.aten.add.", "invalid overload name");
    expect_target_error("torch.ops.aten.add.Tensor.extra", "invalid operator name");
    expect_target_error("torch.ops.aten.add.Tensor-1", "invalid overload name");
}

static void test_torchvision_arguments()
{
    const pnnx::ExportedArgument input = make_tensor("input");
    const pnnx::ExportedArgument weight = make_tensor("weight");
    const pnnx::ExportedArgument offset = make_tensor("offset");
    const pnnx::ExportedArgument mask = make_tensor("mask");
    const pnnx::ExportedArgument bias = make_tensor("bias");
    const pnnx::ExportedArgument rois = make_tensor("rois");

    const std::vector<pnnx::ExportedNamedArgument> deform_inputs = {
        make_input("input", input, pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL),
        make_input("weight", weight, pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL),
        make_input("offset", offset, pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL),
        make_input("mask", mask, pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL),
        make_input("bias", bias, pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL),
        make_input("stride_h", make_int(1), pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL),
        make_input("stride_w", make_int(1), pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL),
        make_input("pad_h", make_int(0), pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL),
        make_input("pad_w", make_int(0), pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL),
        make_input("dilation_h", make_int(1), pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL),
        make_input("dilation_w", make_int(1), pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL),
        make_input("groups", make_int(1), pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL),
        make_input("offset_groups", make_int(1), pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL),
        make_input("use_mask", make_bool(true), pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL),
    };
    const std::vector<pnnx::CanonicalExportedArgument> deform_expected = {
        make_expected("input", input),
        make_expected("weight", weight),
        make_expected("offset", offset),
        make_expected("mask", mask),
        make_expected("bias", bias),
        make_expected("stride_h", make_int(1)),
        make_expected("stride_w", make_int(1)),
        make_expected("pad_h", make_int(0)),
        make_expected("pad_w", make_int(0)),
        make_expected("dilation_h", make_int(1)),
        make_expected("dilation_w", make_int(1)),
        make_expected("groups", make_int(1)),
        make_expected("offset_groups", make_int(1)),
        make_expected("use_mask", make_bool(true)),
    };
    expect_canonical(make_node("torch.ops.torchvision.deform_conv2d.default", deform_inputs), deform_expected, "torchvision deform conv arguments");

    const std::vector<pnnx::ExportedNamedArgument> roi_inputs = {
        make_input("input", input, pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL),
        make_input("rois", rois, pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL),
        make_input("spatial_scale", make_float(0.25), pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL),
        make_input("pooled_height", make_int(3), pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL),
        make_input("pooled_width", make_int(3), pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL),
        make_input("sampling_ratio", make_int(3), pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL),
        make_input("aligned", make_bool(false), pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL),
    };
    const std::vector<pnnx::CanonicalExportedArgument> roi_expected = {
        make_expected("input", input),
        make_expected("rois", rois),
        make_expected("spatial_scale", make_float(0.25)),
        make_expected("pooled_height", make_int(3)),
        make_expected("pooled_width", make_int(3)),
        make_expected("sampling_ratio", make_int(3)),
        make_expected("aligned", make_bool(false)),
    };
    expect_canonical(make_node("torch.ops.torchvision.roi_align.default", roi_inputs), roi_expected, "torchvision roi align arguments");

    expect_canonical_error(make_node("torch.ops.custom.foo.default", std::vector<pnnx::ExportedNamedArgument>()), make_header(), "unsupported exported operator", "unknown custom operator");

    std::vector<pnnx::ExportedNamedArgument> invalid_roi_inputs = roi_inputs;
    invalid_roi_inputs[2].arg = make_int(1);
    expect_canonical_error(make_node("torch.ops.torchvision.roi_align.default", invalid_roi_inputs), make_header(), "spatial_scale", "torchvision argument type");
}

static void test_default_arguments()
{
    const pnnx::ExportedArgument x = make_tensor("x");
    const pnnx::ExportedArgument weight = make_tensor("p_weight");

    expect_canonical(
        make_node("torch.ops.aten.linear.default", std::vector<pnnx::ExportedNamedArgument>{
                                                       make_input("input", x, pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL),
                                                       make_input("weight", weight, pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL)}),
        std::vector<pnnx::CanonicalExportedArgument>{make_expected("input", x), make_expected("weight", weight), make_expected("bias", make_none())}, "linear defaults");

    expect_canonical(
        make_node("torch.ops.aten.add.Tensor", std::vector<pnnx::ExportedNamedArgument>{
                                                   make_input("self", x, pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL),
                                                   make_input("other", x, pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL)}),
        std::vector<pnnx::CanonicalExportedArgument>{make_expected("self", x), make_expected("other", x), make_expected("alpha", make_int(1))}, "add defaults");

    expect_canonical(
        make_node("torch.ops.aten.flatten.using_ints", std::vector<pnnx::ExportedNamedArgument>{
                                                           make_input("self", x, pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL)}),
        std::vector<pnnx::CanonicalExportedArgument>{make_expected("self", x), make_expected("start_dim", make_int(0)), make_expected("end_dim", make_int(-1))}, "flatten defaults");

    expect_canonical(
        make_node("torch.ops.aten.conv2d.default", std::vector<pnnx::ExportedNamedArgument>{
                                                       make_input("input", x, pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL),
                                                       make_input("weight", weight, pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL)}),
        std::vector<pnnx::CanonicalExportedArgument>{make_expected("input", x), make_expected("weight", weight), make_expected("bias", make_none()), make_expected("stride", make_ints(std::vector<int64_t>{1, 1})), make_expected("padding", make_ints(std::vector<int64_t>{0, 0})), make_expected("dilation", make_ints(std::vector<int64_t>{1, 1})), make_expected("groups", make_int(1))}, "conv2d defaults");

    expect_canonical(
        make_node("torch.ops.aten.isclose.default", std::vector<pnnx::ExportedNamedArgument>{
                                                        make_input("self", x, pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL),
                                                        make_input("other", x, pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL)}),
        std::vector<pnnx::CanonicalExportedArgument>{make_expected("self", x), make_expected("other", x), make_expected("rtol", make_float(1.0000000000000001e-05)), make_expected("atol", make_float(1e-08)), make_expected("equal_nan", make_bool(false))}, "float bool defaults");

    expect_canonical(
        make_node("torch.ops.aten.gelu.default", std::vector<pnnx::ExportedNamedArgument>{
                                                     make_input("self", x, pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL)}),
        std::vector<pnnx::CanonicalExportedArgument>{make_expected("self", x), make_expected("approximate", make_string("none"))}, "string defaults");

    expect_canonical(
        make_node("torch.ops.aten.sub.Tensor", std::vector<pnnx::ExportedNamedArgument>{
                                                   make_input("self", x, pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL),
                                                   make_input("other", make_complex(0.0, 4.0), pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL)}),
        std::vector<pnnx::CanonicalExportedArgument>{make_expected("self", x), make_expected("other", make_complex(0.0, 4.0)), make_expected("alpha", make_int(1))}, "complex scalar");

    expect_canonical(
        make_node("torch.ops.aten.max_pool2d.default", std::vector<pnnx::ExportedNamedArgument>{
                                                           make_input("self", x, pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL),
                                                           make_input("kernel_size", make_ints(std::vector<int64_t>{2, 2}), pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL)}),
        std::vector<pnnx::CanonicalExportedArgument>{make_expected("self", x), make_expected("kernel_size", make_ints(std::vector<int64_t>{2, 2})), make_expected("stride", make_ints(std::vector<int64_t>())), make_expected("padding", make_ints(std::vector<int64_t>{0, 0})), make_expected("dilation", make_ints(std::vector<int64_t>{1, 1})), make_expected("ceil_mode", make_bool(false))}, "empty list defaults");

    expect_canonical(
        make_node("torch.ops.aten.triu_indices.default", std::vector<pnnx::ExportedNamedArgument>{
                                                             make_input("row", make_int(3), pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL),
                                                             make_input("col", make_int(4), pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL)}),
        std::vector<pnnx::CanonicalExportedArgument>{make_expected("row", make_int(3)), make_expected("col", make_int(4)), make_expected("offset", make_int(0)), make_expected("dtype", make_enum(pnnx::EXPORTED_ARGUMENT_SCALAR_TYPE, 5)), make_expected("layout", make_none()), make_expected("device", make_none()), make_expected("pin_memory", make_none())}, "scalar type default");

    expect_canonical(
        make_node("torch.ops.aten.contiguous.default", std::vector<pnnx::ExportedNamedArgument>{
                                                           make_input("self", x, pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL)}),
        std::vector<pnnx::CanonicalExportedArgument>{make_expected("self", x), make_expected("memory_format", make_enum(pnnx::EXPORTED_ARGUMENT_MEMORY_FORMAT, 1))}, "memory format default");
}

static void test_binding_order_and_legacy_kind()
{
    const pnnx::ExportedArgument x = make_tensor("x");
    const pnnx::ExportedArgument weight = make_tensor("p_weight");

    expect_canonical(
        make_node("torch.ops.aten.linear.default", std::vector<pnnx::ExportedNamedArgument>{
                                                       make_input("weight", weight, pnnx::EXPORTED_ARGUMENT_KIND_KEYWORD),
                                                       make_input("input", x, pnnx::EXPORTED_ARGUMENT_KIND_KEYWORD)}),
        std::vector<pnnx::CanonicalExportedArgument>{make_expected("input", x), make_expected("weight", weight), make_expected("bias", make_none())}, "keyword reorder");

    expect_canonical(
        make_node("torch.ops.aten.add.Tensor", std::vector<pnnx::ExportedNamedArgument>{
                                                   make_input("self", x, pnnx::EXPORTED_ARGUMENT_KIND_MISSING),
                                                   make_input("other", x, pnnx::EXPORTED_ARGUMENT_KIND_MISSING)}),
        std::vector<pnnx::CanonicalExportedArgument>{make_expected("self", x), make_expected("other", x), make_expected("alpha", make_int(1))}, "legacy missing kind");

    expect_canonical(
        make_node("torch.ops.aten.add.Tensor", std::vector<pnnx::ExportedNamedArgument>{
                                                   make_input("self", x, pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL),
                                                   make_input("other", x, pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL),
                                                   make_input("alpha", make_int(3), pnnx::EXPORTED_ARGUMENT_KIND_KEYWORD)}),
        std::vector<pnnx::CanonicalExportedArgument>{make_expected("self", x), make_expected("other", x), make_expected("alpha", make_int(3))}, "explicit keyword");

    expect_canonical(
        make_node("torch.ops.aten.flatten.using_ints", std::vector<pnnx::ExportedNamedArgument>{
                                                           make_input("self", x, pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL),
                                                           make_input("end_dim", make_int(-2), pnnx::EXPORTED_ARGUMENT_KIND_KEYWORD)}),
        std::vector<pnnx::CanonicalExportedArgument>{make_expected("self", x), make_expected("start_dim", make_int(0)), make_expected("end_dim", make_int(-2))}, "skip default before keyword");
}

static void test_binding_errors()
{
    const pnnx::ExportedArgument x = make_tensor("x");

    expect_canonical_error(
        make_node("torch.ops.aten.add.Tensor", std::vector<pnnx::ExportedNamedArgument>{
                                                   make_input("self", x, pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL),
                                                   make_input("self", x, pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL)}),
        make_header(), "duplicate argument self", "duplicate argument");

    expect_canonical_error(
        make_node("torch.ops.aten.add.Tensor", std::vector<pnnx::ExportedNamedArgument>{
                                                   make_input("self", x, pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL),
                                                   make_input("unknown", x, pnnx::EXPORTED_ARGUMENT_KIND_KEYWORD)}),
        make_header(), "unknown argument unknown", "unknown argument");

    expect_canonical_error(
        make_node("torch.ops.aten.add.Tensor", std::vector<pnnx::ExportedNamedArgument>{
                                                   make_input("self", x, pnnx::EXPORTED_ARGUMENT_KIND_UNKNOWN),
                                                   make_input("other", x, pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL)}),
        make_header(), "unknown argument kind", "unknown kind");

    expect_canonical_error(
        make_node("torch.ops.aten.add.Tensor", std::vector<pnnx::ExportedNamedArgument>{
                                                   make_input("other", x, pnnx::EXPORTED_ARGUMENT_KIND_KEYWORD),
                                                   make_input("self", x, pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL)}),
        make_header(), "positional argument self follows a keyword", "positional after keyword");

    expect_canonical_error(
        make_node("torch.ops.aten.add.Tensor", std::vector<pnnx::ExportedNamedArgument>{
                                                   make_input("self", x, pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL),
                                                   make_input("other", x, pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL),
                                                   make_input("alpha", make_int(2), pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL)}),
        make_header(), "keyword-only argument alpha", "keyword only positional");

    expect_canonical_error(
        make_node("torch.ops.aten.add.Tensor", std::vector<pnnx::ExportedNamedArgument>{
                                                   make_input("other", x, pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL),
                                                   make_input("self", x, pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL)}),
        make_header(), "expected positional argument self", "positional order");

    expect_canonical_error(
        make_node("torch.ops.aten.linear.default", std::vector<pnnx::ExportedNamedArgument>{
                                                       make_input("input", x, pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL)}),
        make_header(), "missing required argument weight", "missing required");

    expect_canonical_error(
        make_node("torch.ops.aten.add.Tensor", std::vector<pnnx::ExportedNamedArgument>{
                                                   make_input("self", x, pnnx::EXPORTED_ARGUMENT_KIND_MISSING),
                                                   make_input("other", x, pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL)}),
        make_header(), "mixes legacy and explicit argument kinds", "mixed kind versions");

    expect_canonical_error(
        make_node("torch.ops.aten.add.Tensor", std::vector<pnnx::ExportedNamedArgument>{
                                                   make_input("self", make_unsupported("as_int_lists"), pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL),
                                                   make_input("other", x, pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL)}),
        make_header(), "unsupported serialized argument as_int_lists", "unsupported argument");

    expect_canonical_error(make_node("torch.ops.aten.no_such_operator.default", std::vector<pnnx::ExportedNamedArgument>()), make_header(), "cannot resolve dispatcher schema", "missing dispatcher schema");
    expect_canonical_error(make_node("torch.ops.aten.relu.default", std::vector<pnnx::ExportedNamedArgument>{make_input("self", x, pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL)}), make_header(-1), "missing aten opset", "missing aten opset");
    expect_canonical_error(make_node("torch.ops.aten.relu.default", std::vector<pnnx::ExportedNamedArgument>{make_input("self", x, pnnx::EXPORTED_ARGUMENT_KIND_POSITIONAL)}), make_header(9), "does not match linked libtorch", "opset mismatch");
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

static std::string argument_text(const pnnx::ExportedArgument& value)
{
    std::ostringstream text;
    if (value.type == pnnx::EXPORTED_ARGUMENT_NONE)
        return "none";
    if (value.type == pnnx::EXPORTED_ARGUMENT_TENSOR)
        return "tensor:" + value.name;
    if (value.type == pnnx::EXPORTED_ARGUMENT_INT)
    {
        text << "int:" << value.int_value;
        return text.str();
    }
    if (value.type == pnnx::EXPORTED_ARGUMENT_INT_LIST)
        return "ints:" + join_ints(value.int_values);
    if (value.type == pnnx::EXPORTED_ARGUMENT_FLOAT)
    {
        text.precision(17);
        text << "float:" << value.float_value;
        return text.str();
    }
    if (value.type == pnnx::EXPORTED_ARGUMENT_BOOL)
        return value.bool_value ? "bool:true" : "bool:false";
    if (value.type == pnnx::EXPORTED_ARGUMENT_COMPLEX)
    {
        text.precision(17);
        text << "complex:" << value.complex_real_value << ',' << value.complex_imag_value;
        return text.str();
    }
    if (value.type == pnnx::EXPORTED_ARGUMENT_STRING)
        return "string:" + value.string_value;

    text << "type:" << value.type;
    return text.str();
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

    for (size_t i = 0; i < program.graph.nodes.size(); i++)
    {
        const pnnx::ExportedNode& node = program.graph.nodes[i];
        pnnx::ExportedOperatorTarget target;
        if (pnnx::parse_exported_operator_target(node.target, target, error) != 0)
        {
            fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }

        std::vector<pnnx::CanonicalExportedArgument> arguments;
        if (pnnx::canonicalize_exported_arguments(node, program.header, arguments, error) != 0)
        {
            fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }

        fprintf(stdout, "node|%s|%s|%s|", node.target.c_str(), target.operator_name.c_str(), target.overload_name.empty() ? "default" : target.overload_name.c_str());
        for (size_t j = 0; j < arguments.size(); j++)
        {
            if (j != 0)
                fprintf(stdout, ";");
            fprintf(stdout, "%s=%s", arguments[j].name.c_str(), argument_text(arguments[j].value).c_str());
        }
        fprintf(stdout, "\n");
    }

    return 0;
}

int main(int argc, char** argv)
{
    if (argc == 3 && std::string(argv[1]) == "inspect")
        return inspect_package(argv[2]);
    if (argc != 1)
        return 2;

    test_targets();
    test_torchvision_arguments();
    test_default_arguments();
    test_binding_order_and_legacy_kind();
    test_binding_errors();

    if (test_failures != 0)
    {
        fprintf(stderr, "%d exported program operator tests failed across %d paths\n", test_failures, test_paths);
        return 1;
    }

    fprintf(stdout, "exported program operator tests passed %d paths\n", test_paths);
    return 0;
}
