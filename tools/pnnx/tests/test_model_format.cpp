// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "model_format.h"

#include <stdio.h>

#include <string>

int main(int argc, char** argv)
{
    if (argc != 3 || std::string(argv[1]) != "detect")
        return 2;

    pnnx::ModelFormatInfo info;
    std::string error;
    if (pnnx::detect_model_format(argv[2], info, error) != 0)
    {
        fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }

    if (info.format == pnnx::MODEL_FORMAT_UNKNOWN)
    {
        fprintf(stdout, "unknown\n");
        return 0;
    }

    if (info.format == pnnx::MODEL_FORMAT_TORCHSCRIPT)
    {
        fprintf(stdout, "torchscript\n");
        return 0;
    }

    if (info.format == pnnx::MODEL_FORMAT_EXPORTED_PROGRAM_PT2)
    {
        fprintf(stdout, "pt2|%s|%llu\n", info.archive_root.c_str(), (unsigned long long)info.archive_version);
        return 0;
    }

    return 2;
}
