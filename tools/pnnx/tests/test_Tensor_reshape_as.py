# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import check_numerical_result, convert_and_import

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x = x.reshape_as(y)
        y = y.reshape_as(z)
        z = z.reshape_as(x)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 16)
    y = torch.rand(6, 2, 2, 2)
    z = torch.rand(48)

    a = net(x, y, z)

    mod = convert_and_import(
        net,
        (x, y, z),
        "test_Tensor_reshape_as",
        pnnx_args=("inputshape=[1,3,16],[6,2,2,2],[48]",),
    )
    if mod is None:
        return True

    b = mod.test_inference()

    passed = True
    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            passed = False
    return check_numerical_result("test_Tensor_reshape_as", passed)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
