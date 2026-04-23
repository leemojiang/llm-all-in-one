import torch 
from torch import nn


def register_debug_hooks(model: nn.Module) -> list:
    """
    在模型的每一层注册 forward hook,用于调试时打印输入输出的 shape。

    Return:
        Handle list

    #Usage
    net = MyModel()
    register_debug_hooks(net)
    dummy_input = torch.randn(1, 3, 224, 224)
    _ = net(dummy_input)
    """

    def hook_fn(module, input, output):
        # 打印模块类型和输入输出 shape
        module_name = module.__class__.__name__
        print(f"[{module_name}]")
        if isinstance(input, tuple):
            print(
                f"  Input shape: {[i.shape for i in input if isinstance(i, torch.Tensor)]}"
            )
        else:
            print(f"  Input shape: {input.shape}")
        if isinstance(output, tuple):
            print(
                f"  Output shape: {[o.shape for o in output if isinstance(o, torch.Tensor)]}"
            )
        else:
            print(f"  Output shape: {output.shape}")
        print("-" * 40)
    handles = []
    for name, module in model.named_modules():
        # 跳过容器模块（如 Sequential）或空模块
        if len(list(module.children())) == 0:
            h = module.register_forward_hook(hook_fn)
            handles.append(h)

    return handles