# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import check_numerical_result, convert_and_import

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w):
        out0 = torch.cat((x, y), dim=1)
        out1 = torch.cat((z, w), dim=3)
        out2 = torch.cat((w, w), dim=2)
        return out0, out1, out2

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 16)
    y = torch.rand(1, 2, 16)
    z = torch.rand(1, 5, 9, 11)
    w = torch.rand(1, 5, 9, 3)

    a0, a1, a2 = net(x, y, z, w)

    mod = convert_and_import(
        net,
        (x, y, z, w),
        "test_torch_cat",
        pnnx_args=("inputshape=[1,3,16],[1,2,16],[1,5,9,11],[1,5,9,3]",),
    )
    if mod is None:
        return True

    b0, b1, b2 = mod.test_inference()

    passed = torch.equal(a0, b0) and torch.equal(a1, b1) and torch.equal(a2, b2)
    return check_numerical_result("test_torch_cat", passed)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
