// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "exported_program_tensor.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <limits>
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

static pnnx::ExportedTensorMeta make_meta(int64_t dtype, const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides, int64_t storage_offset)
{
    pnnx::ExportedTensorMeta meta;
    meta.dtype = dtype;
    for (size_t i = 0; i < sizes.size(); i++)
        meta.sizes.push_back(pnnx::ExportedSymInt(sizes[i]));
    for (size_t i = 0; i < strides.size(); i++)
        meta.strides.push_back(pnnx::ExportedSymInt(strides[i]));
    meta.storage_offset = storage_offset;
    meta.layout = 7;
    meta.device_type = "cpu";
    return meta;
}

static void append_bits(std::vector<char>& data, uint64_t bits, size_t width, pnnx::Pt2ByteOrder byte_order)
{
    for (size_t i = 0; i < width; i++)
    {
        const size_t shift_index = byte_order == pnnx::PT2_BYTE_ORDER_LITTLE ? i : width - 1 - i;
        data.push_back((char)((bits >> (shift_index * 8)) & 0xff));
    }
}

static void append_float(std::vector<char>& data, float value, pnnx::Pt2ByteOrder byte_order)
{
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    append_bits(data, bits, sizeof(bits), byte_order);
}

static void append_int32(std::vector<char>& data, int32_t value, pnnx::Pt2ByteOrder byte_order)
{
    append_bits(data, (uint32_t)value, sizeof(value), byte_order);
}

template<typename T>
static std::vector<T> read_native_values(const std::vector<char>& data)
{
    std::vector<T> values;
    if (data.size() % sizeof(T) != 0)
        return values;

    values.resize(data.size() / sizeof(T));
    for (size_t i = 0; i < values.size(); i++)
        memcpy(&values[i], &data[i * sizeof(T)], sizeof(T));
    return values;
}

static bool same_shape(const std::vector<int>& actual, const std::vector<int>& expected)
{
    return actual == expected;
}

static bool same_floats(const std::vector<float>& actual, const std::vector<float>& expected)
{
    if (actual.size() != expected.size())
        return false;

    for (size_t i = 0; i < actual.size(); i++)
    {
        if (actual[i] != expected[i])
            return false;
    }
    return true;
}

static bool expect_success(const pnnx::ExportedTensorMeta& meta, const std::vector<char>& storage, pnnx::Pt2ByteOrder byte_order, pnnx::MaterializedExportedTensor& tensor, const char* name)
{
    std::string error = "stale";
    if (pnnx::materialize_exported_tensor(meta, storage, byte_order, tensor, error) != 0)
    {
        check(false, name, "unexpected error " + error);
        return false;
    }

    check(error.empty(), name, "success did not clear error");
    return true;
}

static void expect_error(const pnnx::ExportedTensorMeta& meta, const std::vector<char>& storage, pnnx::Pt2ByteOrder byte_order, const std::string& expected_error, const char* name)
{
    pnnx::MaterializedExportedTensor tensor;
    tensor.pnnx_type = 99;
    tensor.shape.push_back(99);
    tensor.data.push_back(99);
    std::string error = "stale";

    const int result = pnnx::materialize_exported_tensor(meta, storage, byte_order, tensor, error);
    check(result != 0, name, "materialization unexpectedly succeeded");
    check(error.find(expected_error) != std::string::npos, name, "unexpected error " + error);
    check(tensor.pnnx_type == 0 && tensor.shape.empty() && tensor.data.empty(), name, "failure retained partial output");
}

static std::vector<char> make_float_storage(size_t count, pnnx::Pt2ByteOrder byte_order)
{
    std::vector<char> storage;
    for (size_t i = 0; i < count; i++)
        append_float(storage, (float)i, byte_order);
    return storage;
}

static void test_contiguous_and_strided_views()
{
    const std::vector<char> storage = make_float_storage(12, pnnx::PT2_BYTE_ORDER_LITTLE);
    pnnx::MaterializedExportedTensor tensor;

    pnnx::ExportedTensorMeta meta = make_meta(7, std::vector<int64_t>{2, 3}, std::vector<int64_t>{3, 1}, 0);
    if (expect_success(meta, storage, pnnx::PT2_BYTE_ORDER_LITTLE, tensor, "contiguous f32"))
    {
        check(tensor.pnnx_type == 1, "contiguous f32", "wrong pnnx type");
        check(same_shape(tensor.shape, std::vector<int>{2, 3}), "contiguous f32", "wrong shape");
        check(same_floats(read_native_values<float>(tensor.data), std::vector<float>{0.f, 1.f, 2.f, 3.f, 4.f, 5.f}), "contiguous f32", "wrong values");
    }

    meta = make_meta(7, std::vector<int64_t>{2, 3}, std::vector<int64_t>{1, 2}, 0);
    if (expect_success(meta, storage, pnnx::PT2_BYTE_ORDER_LITTLE, tensor, "transposed f32"))
        check(same_floats(read_native_values<float>(tensor.data), std::vector<float>{0.f, 2.f, 4.f, 1.f, 3.f, 5.f}), "transposed f32", "wrong values");

    meta = make_meta(7, std::vector<int64_t>{2}, std::vector<int64_t>{1}, 3);
    if (expect_success(meta, storage, pnnx::PT2_BYTE_ORDER_LITTLE, tensor, "offset f32"))
        check(same_floats(read_native_values<float>(tensor.data), std::vector<float>{3.f, 4.f}), "offset f32", "wrong values");

    const pnnx::ExportedTensorMeta left = make_meta(7, std::vector<int64_t>{3}, std::vector<int64_t>{1}, 0);
    const pnnx::ExportedTensorMeta right = make_meta(7, std::vector<int64_t>{3}, std::vector<int64_t>{1}, 2);
    if (expect_success(left, storage, pnnx::PT2_BYTE_ORDER_LITTLE, tensor, "shared storage left"))
        check(same_floats(read_native_values<float>(tensor.data), std::vector<float>{0.f, 1.f, 2.f}), "shared storage left", "wrong values");
    if (expect_success(right, storage, pnnx::PT2_BYTE_ORDER_LITTLE, tensor, "shared storage right"))
        check(same_floats(read_native_values<float>(tensor.data), std::vector<float>{2.f, 3.f, 4.f}), "shared storage right", "wrong values");

    meta = make_meta(7, std::vector<int64_t>(), std::vector<int64_t>(), 4);
    if (expect_success(meta, storage, pnnx::PT2_BYTE_ORDER_LITTLE, tensor, "scalar f32"))
    {
        check(tensor.shape.empty(), "scalar f32", "scalar shape changed");
        check(same_floats(read_native_values<float>(tensor.data), std::vector<float>{4.f}), "scalar f32", "wrong value");
    }

    meta = make_meta(7, std::vector<int64_t>{0, 3}, std::vector<int64_t>{3, 1}, 0);
    if (expect_success(meta, std::vector<char>(), pnnx::PT2_BYTE_ORDER_LITTLE, tensor, "zero element tensor"))
    {
        check(same_shape(tensor.shape, std::vector<int>{0, 3}), "zero element tensor", "wrong shape");
        check(tensor.data.empty(), "zero element tensor", "zero tensor has data");
    }
}

static void test_dtype_mapping()
{
    struct DtypeCase
    {
        int64_t exported_type;
        int pnnx_type;
        size_t element_size;
    };

    const DtypeCase cases[] = {
        {1, 8, 1},
        {2, 7, 1},
        {3, 6, 2},
        {4, 4, 4},
        {5, 5, 8},
        {6, 3, 2},
        {7, 1, 4},
        {8, 2, 8},
        {9, 12, 4},
        {10, 10, 8},
        {11, 11, 16},
        {12, 9, 1},
        {13, 13, 2},
    };

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
    {
        const pnnx::ExportedTensorMeta meta = make_meta(cases[i].exported_type, std::vector<int64_t>{1}, std::vector<int64_t>{1}, 0);
        const std::vector<char> storage(cases[i].element_size, 0);
        pnnx::MaterializedExportedTensor tensor;
        if (expect_success(meta, storage, pnnx::PT2_BYTE_ORDER_LITTLE, tensor, "dtype mapping"))
        {
            check(tensor.pnnx_type == cases[i].pnnx_type, "dtype mapping", "wrong pnnx type");
            check(tensor.data.size() == cases[i].element_size, "dtype mapping", "wrong element size");
        }
    }
}

static void test_byte_order()
{
    std::vector<char> storage;
    append_int32(storage, 0x01020304, pnnx::PT2_BYTE_ORDER_BIG);
    append_int32(storage, -2, pnnx::PT2_BYTE_ORDER_BIG);

    pnnx::MaterializedExportedTensor tensor;
    const pnnx::ExportedTensorMeta int_meta = make_meta(4, std::vector<int64_t>{2}, std::vector<int64_t>{1}, 0);
    if (expect_success(int_meta, storage, pnnx::PT2_BYTE_ORDER_BIG, tensor, "big endian i32"))
    {
        const std::vector<int32_t> values = read_native_values<int32_t>(tensor.data);
        check(values.size() == 2 && values[0] == 0x01020304 && values[1] == -2, "big endian i32", "wrong values");
    }

    storage.clear();
    append_float(storage, 1.f, pnnx::PT2_BYTE_ORDER_BIG);
    append_float(storage, 2.f, pnnx::PT2_BYTE_ORDER_BIG);
    append_float(storage, 3.f, pnnx::PT2_BYTE_ORDER_BIG);
    append_float(storage, 4.f, pnnx::PT2_BYTE_ORDER_BIG);

    const pnnx::ExportedTensorMeta complex_meta = make_meta(10, std::vector<int64_t>{2}, std::vector<int64_t>{1}, 0);
    if (expect_success(complex_meta, storage, pnnx::PT2_BYTE_ORDER_BIG, tensor, "big endian complex64"))
        check(same_floats(read_native_values<float>(tensor.data), std::vector<float>{1.f, 2.f, 3.f, 4.f}), "big endian complex64", "complex components changed");

    storage = make_float_storage(6, pnnx::PT2_BYTE_ORDER_BIG);
    const pnnx::ExportedTensorMeta transposed_meta = make_meta(7, std::vector<int64_t>{2, 3}, std::vector<int64_t>{1, 2}, 0);
    if (expect_success(transposed_meta, storage, pnnx::PT2_BYTE_ORDER_BIG, tensor, "big endian transposed f32"))
        check(same_floats(read_native_values<float>(tensor.data), std::vector<float>{0.f, 2.f, 4.f, 1.f, 3.f, 5.f}), "big endian transposed f32", "wrong values");
}

static void test_errors()
{
    const std::vector<char> storage = make_float_storage(6, pnnx::PT2_BYTE_ORDER_LITTLE);

    expect_error(make_meta(0, std::vector<int64_t>{1}, std::vector<int64_t>{1}, 0), storage, pnnx::PT2_BYTE_ORDER_LITTLE, "unsupported exported tensor dtype", "unknown dtype");
    expect_error(make_meta(28, std::vector<int64_t>{1}, std::vector<int64_t>{1}, 0), storage, pnnx::PT2_BYTE_ORDER_LITTLE, "unsupported exported tensor dtype", "unsupported uint16 dtype");
    expect_error(make_meta(7, std::vector<int64_t>{-1}, std::vector<int64_t>{1}, 0), storage, pnnx::PT2_BYTE_ORDER_LITTLE, "negative tensor size", "negative size");
    expect_error(make_meta(7, std::vector<int64_t>{(int64_t)INT_MAX + 1}, std::vector<int64_t>{1}, 0), storage, pnnx::PT2_BYTE_ORDER_LITTLE, "does not fit pnnx shape", "shape int overflow");
    expect_error(make_meta(7, std::vector<int64_t>{2}, std::vector<int64_t>(), 0), storage, pnnx::PT2_BYTE_ORDER_LITTLE, "stride rank does not match", "stride rank mismatch");
    expect_error(make_meta(7, std::vector<int64_t>{2}, std::vector<int64_t>{-1}, 0), storage, pnnx::PT2_BYTE_ORDER_LITTLE, "negative tensor stride", "negative stride");
    expect_error(make_meta(7, std::vector<int64_t>{2}, std::vector<int64_t>{1}, -1), storage, pnnx::PT2_BYTE_ORDER_LITTLE, "negative storage offset", "negative storage offset");

    pnnx::ExportedTensorMeta meta = make_meta(7, std::vector<int64_t>{2}, std::vector<int64_t>{1}, 0);
    meta.layout = 1;
    expect_error(meta, storage, pnnx::PT2_BYTE_ORDER_LITTLE, "unsupported tensor layout", "sparse layout");

    meta = make_meta(7, std::vector<int64_t>{2}, std::vector<int64_t>{1}, 0);
    meta.sizes[0].type = pnnx::EXPORTED_SYM_INT_EXPRESSION;
    meta.sizes[0].expression = "Symbol('u0', integer=True)";
    expect_error(meta, storage, pnnx::PT2_BYTE_ORDER_LITTLE, "symbolic tensor size at dimension 0 cannot be materialized", "symbolic state size");

    meta = make_meta(7, std::vector<int64_t>{2}, std::vector<int64_t>{1}, 0);
    meta.strides[0].type = pnnx::EXPORTED_SYM_INT_EXPRESSION;
    meta.strides[0].expression = "Symbol('u0', integer=True)";
    expect_error(meta, storage, pnnx::PT2_BYTE_ORDER_LITTLE, "symbolic tensor stride at dimension 0 cannot be materialized", "symbolic state stride");

    meta = make_meta(7, std::vector<int64_t>{2}, std::vector<int64_t>{1}, 0);
    meta.storage_offset.type = pnnx::EXPORTED_SYM_INT_EXPRESSION;
    meta.storage_offset.expression = "Symbol('u0', integer=True)";
    expect_error(meta, storage, pnnx::PT2_BYTE_ORDER_LITTLE, "symbolic storage offset cannot be materialized", "symbolic state offset");

    expect_error(make_meta(7, std::vector<int64_t>{1}, std::vector<int64_t>{1}, 0), std::vector<char>(3, 0), pnnx::PT2_BYTE_ORDER_LITTLE, "not aligned to element size", "unaligned storage");
    expect_error(make_meta(7, std::vector<int64_t>{1}, std::vector<int64_t>{1}, 6), storage, pnnx::PT2_BYTE_ORDER_LITTLE, "tensor view exceeds storage", "offset out of bounds");
    expect_error(make_meta(7, std::vector<int64_t>{2, 2}, std::vector<int64_t>{5, 1}, 0), storage, pnnx::PT2_BYTE_ORDER_LITTLE, "tensor view exceeds storage", "strided view out of bounds");
    expect_error(make_meta(7, std::vector<int64_t>{2, 3}, std::vector<int64_t>{0, 1}, 0), make_float_storage(3, pnnx::PT2_BYTE_ORDER_LITTLE), pnnx::PT2_BYTE_ORDER_LITTLE, "overlapping tensor view expands storage", "broadcast view expansion");
    expect_error(make_meta(7, std::vector<int64_t>{46341, 46341}, std::vector<int64_t>{0, 0}, 0), make_float_storage(1, pnnx::PT2_BYTE_ORDER_LITTLE), pnnx::PT2_BYTE_ORDER_LITTLE, "tensor element count does not fit pnnx attribute", "pnnx element count overflow");
    expect_error(make_meta(7, std::vector<int64_t>{INT_MAX, INT_MAX, INT_MAX}, std::vector<int64_t>{0, 0, 0}, 0), storage, pnnx::PT2_BYTE_ORDER_LITTLE, "tensor element count overflow", "element count overflow");
    expect_error(make_meta(7, std::vector<int64_t>{INT_MAX}, std::vector<int64_t>{std::numeric_limits<int64_t>::max()}, std::numeric_limits<int64_t>::max()), storage, pnnx::PT2_BYTE_ORDER_LITTLE, "tensor view offset overflow", "view offset overflow");
}

static std::string join_shape(const std::vector<int>& shape)
{
    std::ostringstream text;
    for (size_t i = 0; i < shape.size(); i++)
    {
        if (i != 0)
            text << ',';
        text << shape[i];
    }
    return text.str();
}

static std::string hex_data(const std::vector<char>& data)
{
    static const char digits[] = "0123456789abcdef";
    std::string text;
    text.reserve(data.size() * 2);
    for (size_t i = 0; i < data.size(); i++)
    {
        const unsigned char byte = (unsigned char)data[i];
        text.push_back(digits[byte >> 4]);
        text.push_back(digits[byte & 15]);
    }
    return text;
}

static int inspect_payload_config(pnnx::Pt2ArchiveReader& reader, const std::string& config_path, const std::string& payload_directory, const char* config_name)
{
    pnnx::JsonValue config_json;
    std::string error;
    if (reader.read_json(config_path, config_json, error) != 0)
    {
        fprintf(stderr, "%s\n", error.c_str());
        return -1;
    }

    pnnx::ExportedPayloadConfig config;
    pnnx::ExportedSchemaError schema_error;
    if (pnnx::parse_exported_payload_config(config_json, config, schema_error) != 0)
    {
        fprintf(stderr, "%s %s\n", schema_error.path.c_str(), schema_error.message.c_str());
        return -1;
    }

    for (std::map<std::string, pnnx::ExportedPayloadEntry>::const_iterator it = config.entries.begin(); it != config.entries.end(); ++it)
    {
        if (it->second.use_pickle || !it->second.has_tensor_meta)
        {
            fprintf(stderr, "pickled payload is unsupported %s\n", it->first.c_str());
            return -1;
        }

        const std::string payload_path = reader.layout().root + payload_directory + it->second.path_name;
        std::vector<char> storage;
        if (reader.read_blob(payload_path, storage, error) != 0)
        {
            fprintf(stderr, "%s\n", error.c_str());
            return -1;
        }

        pnnx::MaterializedExportedTensor tensor;
        if (pnnx::materialize_exported_tensor(it->second.tensor_meta, storage, reader.layout().byte_order, tensor, error) != 0)
        {
            fprintf(stderr, "%s: %s\n", it->first.c_str(), error.c_str());
            return -1;
        }

        fprintf(stdout, "tensor|%s|%s|%s|%d|%s|%s\n",
                config_name,
                it->first.c_str(),
                it->second.path_name.c_str(),
                tensor.pnnx_type,
                join_shape(tensor.shape).c_str(),
                hex_data(tensor.data).c_str());
    }

    return 0;
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

    if (inspect_payload_config(reader, reader.layout().weights_config_path, "/data/weights/", "weights") != 0)
        return 1;
    if (inspect_payload_config(reader, reader.layout().constants_config_path, "/data/constants/", "constants") != 0)
        return 1;

    return 0;
}

int main(int argc, char** argv)
{
    if (argc == 3 && std::string(argv[1]) == "inspect")
        return inspect_package(argv[2]);
    if (argc != 1)
        return 2;

    test_contiguous_and_strided_views();
    test_dtype_mapping();
    test_byte_order();
    test_errors();

    if (test_failures != 0)
    {
        fprintf(stderr, "%d exported program tensor tests failed across %d paths\n", test_failures, test_paths);
        return 1;
    }

    fprintf(stdout, "exported program tensor tests passed %d paths\n", test_paths);
    return 0;
}
