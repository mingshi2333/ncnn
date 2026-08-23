# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torchvision.models as models
from packaging import version

def test():
    net = models.resnet18().half().float()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 224, 224)

    a = net(x)

    import os
    use_exported_program = version.parse(torch.__version__) >= version.parse('2.9')
    if use_exported_program:
        # export exported program
        ep = torch.export.export(net, (x,))
        torch.export.save(ep, "test_resnet18.pt2")

        # exported program to pnnx
        if os.system("../../src/pnnx test_resnet18.pt2") != 0:
            return False

        # pnnx inference
        import test_resnet18_pnnx
        b = test_resnet18_pnnx.test_inference()
    else:
        # export torchscript
        mod = torch.jit.trace(net, x)
        mod.save("test_resnet18.pt")

        # torchscript to pnnx
        if os.system("../../src/pnnx test_resnet18.pt inputshape=[1,3,224,224]") != 0:
            return False

    import test_resnet18_ncnn
    c = test_resnet18_ncnn.test_inference()

    if use_exported_program:
        return torch.allclose(a, b, 1e-3, 1e-3) and torch.allclose(a, c, 1e-2, 1e-2)

    return torch.allclose(a, c, 1e-2, 1e-2)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
