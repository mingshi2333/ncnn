// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pt2_archive.h"

#include <stdio.h>

#include <string>
#include <vector>

static const char* byte_order_name(pnnx::Pt2ByteOrder byte_order)
{
    if (byte_order == pnnx::PT2_BYTE_ORDER_LITTLE)
        return "little";
    if (byte_order == pnnx::PT2_BYTE_ORDER_BIG)
        return "big";

    return "unknown";
}

static const char* json_type_name(pnnx::JsonType type)
{
    if (type == pnnx::JSON_NULL)
        return "null";
    if (type == pnnx::JSON_BOOL)
        return "bool";
    if (type == pnnx::JSON_INT64)
        return "int64";
    if (type == pnnx::JSON_UINT64)
        return "uint64";
    if (type == pnnx::JSON_DOUBLE)
        return "double";
    if (type == pnnx::JSON_STRING)
        return "string";
    if (type == pnnx::JSON_ARRAY)
        return "array";
    if (type == pnnx::JSON_OBJECT)
        return "object";

    return "unknown";
}

int main(int argc, char** argv)
{
    if (argc < 3)
        return 2;

    pnnx::Pt2ArchiveReader reader;
    std::string error;
    if (reader.open(argv[2], error) != 0)
    {
        fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }

    const std::string command = argv[1];
    if (command == "reopen" && argc == 4)
    {
        const std::string old_model_json_path = reader.layout().model_json_path;
        if (reader.open(argv[3], error) == 0)
        {
            fprintf(stderr, "second package unexpectedly opened\n");
            return 1;
        }

        const std::string open_error = error;
        std::vector<char> data;
        if (reader.read_blob(old_model_json_path, data, error) == 0)
        {
            fprintf(stderr, "stale package remained readable\n");
            return 1;
        }

        fprintf(stdout, "%s|%s\n", open_error.c_str(), error.c_str());
        return 0;
    }

    if (command == "inspect" && argc == 3)
    {
        const pnnx::Pt2PackageLayout& layout = reader.layout();
        fprintf(stdout, "%s|%llu|%s|%s|%s|%s|%s\n",
                layout.root.c_str(),
                (unsigned long long)layout.archive_version,
                layout.model_name.c_str(),
                byte_order_name(layout.byte_order),
                layout.model_json_path.c_str(),
                layout.weights_config_path.c_str(),
                layout.constants_config_path.c_str());
        return 0;
    }

    if (command == "read-json" && argc == 4)
    {
        pnnx::JsonValue value;
        if (reader.read_json(argv[3], value, error) != 0)
        {
            fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }

        fprintf(stdout, "%s\n", json_type_name(value.type()));
        return 0;
    }

    if (command == "read-blob" && argc == 4)
    {
        std::vector<char> data;
        if (reader.read_blob(argv[3], data, error) != 0)
        {
            fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }

        fprintf(stdout, "%llu\n", (unsigned long long)data.size());
        return 0;
    }

    return 2;
}
