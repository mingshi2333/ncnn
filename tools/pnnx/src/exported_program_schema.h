// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_EXPORTED_PROGRAM_SCHEMA_H
#define PNNX_EXPORTED_PROGRAM_SCHEMA_H

#include "json_reader.h"

#include <stdint.h>

#include <map>
#include <string>
#include <vector>

namespace pnnx {

struct ExportedProgramHeader
{
    ExportedProgramHeader();

    int schema_major;
    int schema_minor;
    std::string torch_version;
    std::map<std::string, int64_t> opset_version;
};

struct ExportedTensorMeta
{
    ExportedTensorMeta();

    int64_t dtype;
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
    int64_t storage_offset;
    int64_t layout;
    bool requires_grad;
    std::string device_type;
    int64_t device_index;
    bool has_device_index;
};

struct ExportedPayloadEntry
{
    ExportedPayloadEntry();

    std::string path_name;
    bool is_param;
    bool use_pickle;
    bool has_tensor_meta;
    ExportedTensorMeta tensor_meta;
};

struct ExportedPayloadConfig
{
    std::map<std::string, ExportedPayloadEntry> entries;
};

struct ExportedSchemaError
{
    std::string path;
    std::string message;
};

int parse_exported_program_header(const JsonValue& value, ExportedProgramHeader& header, ExportedSchemaError& error);
int parse_exported_tensor_meta(const JsonValue& value, ExportedTensorMeta& tensor_meta, ExportedSchemaError& error, const std::string& path);
int parse_exported_payload_config(const JsonValue& value, ExportedPayloadConfig& payload_config, ExportedSchemaError& error);

} // namespace pnnx

#endif // PNNX_EXPORTED_PROGRAM_SCHEMA_H
