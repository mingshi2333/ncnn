# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

PASS = "PASS"
EXPORT_UNSUPPORTED = "EXPORT_UNSUPPORTED"
PT2_FRONTEND_UNSUPPORTED = "PT2_FRONTEND_UNSUPPORTED"
PNNX_LOWERING_UNSUPPORTED = "PNNX_LOWERING_UNSUPPORTED"
NUMERICAL_MISMATCH = "NUMERICAL_MISMATCH"
UNCLASSIFIED = "UNCLASSIFIED"


# Each pt2 test must have an explicit expected category.  A non-PASS entry must
# also provide a stable diagnostic substring so that unrelated regressions do
# not become expected failures accidentally.
PT2_EXPECTATIONS = {
    "test_F_conv2d": (PT2_FRONTEND_UNSUPPORTED, "unsupported exported operator torch.ops.aten.conv2d.padding"),
    "test_F_linear": (PASS, ""),
    "test_F_relu": (PT2_FRONTEND_UNSUPPORTED, "unsupported exported operator torch.ops.aten.mul.Tensor"),
    "test_Tensor_reshape": (PT2_FRONTEND_UNSUPPORTED, "unsupported exported operator torch.ops.aten.reshape.default"),
    "test_nn_Conv2d": (PT2_FRONTEND_UNSUPPORTED, "unsupported exported operator torch.ops.aten.conv2d.padding"),
    "test_nn_Linear": (PT2_FRONTEND_UNSUPPORTED, "unsupported exported operator torch.ops.aten._weight_norm.default"),
    "test_torch_flatten": (PASS, ""),
}
