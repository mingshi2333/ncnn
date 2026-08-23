// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_EXPORTED_PROGRAM_OPERATOR_H
#define PNNX_EXPORTED_PROGRAM_OPERATOR_H

#include "exported_program_schema.h"

#include <string>
#include <vector>

namespace pnnx {

struct ExportedAtenTarget
{
    std::string operator_name;
    std::string overload_name;
};

struct CanonicalExportedArgument
{
    std::string name;
    ExportedArgument value;
};

int parse_exported_aten_target(const std::string& target, ExportedAtenTarget& result, std::string& error);

int canonicalize_exported_arguments(const ExportedNode& node,
                                    const ExportedProgramHeader& header,
                                    std::vector<CanonicalExportedArgument>& result,
                                    std::string& error);

bool is_exported_aten_target_supported(const ExportedAtenTarget& target);

} // namespace pnnx

#endif // PNNX_EXPORTED_PROGRAM_OPERATOR_H
