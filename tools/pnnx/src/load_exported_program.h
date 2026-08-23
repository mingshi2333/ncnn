// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_LOAD_EXPORTED_PROGRAM_H
#define PNNX_LOAD_EXPORTED_PROGRAM_H

#include "exported_program_schema.h"
#include "exported_program_tensor.h"
#include "ir.h"

#include <map>
#include <string>

namespace pnnx {

int lower_exported_program(const ExportedProgram& program,
                           const std::map<std::string, MaterializedExportedTensor>& state,
                           Graph& graph,
                           std::string& error);

int load_exported_program(const std::string& pt2path, Graph& graph, std::string& error);

} // namespace pnnx

#endif // PNNX_LOAD_EXPORTED_PROGRAM_H
