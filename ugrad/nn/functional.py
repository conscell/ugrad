from ..tensorbase import TensorBase, inf
from ..tensor import Tensor, Node

"""Functional interface"""


def exp(input):
    return input.exp()


def log(input):
    return input.log()


def relu(input):
    return input.relu()


def sigmoid(input):
    return input.sigmoid()


def tanh(input):
    return input.tanh()


def log_softmax(input, dim=-1):
    return input.log_softmax(dim=dim)


def binary_cross_entropy(input, target):
    return -(target * input.log() + (1 - target) * (1 - input).log()).sum() / target.shape[0]


def nll_loss(input, target):
    return -(input * target).sum() / target.shape[0]


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1):
    N, C, H, W = input.shape
    F, _, HH, WW = weight.shape
    dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding

    if padding[0] or padding[1]:
        H += 2 * padding[0]
        W += 2 * padding[1]
        x = TensorBase.zeros((N, C, H, W), device=input.device)
        x[...,padding[0] : -padding[0], padding[1] : -padding[1]] = input.data
    else:
        x = input.data

    Ho = (H - dilation[0] * (HH - 1) - 1) // stride[0] + 1
    Wo = (W - dilation[1] * (WW - 1) - 1) // stride[1] + 1

    shape = (C, HH, WW, N, Ho, Wo)
    strides = (H * W, dilation[0] * W, dilation[1], C * H * W, W * stride[0], stride[1])

    # (N, C, H, W) -> (C, HH, WW, N, Ho, Wo) -> (C * HH * WW, N * Ho * Wo)
    x_strided = TensorBase.as_strided(x, shape=shape, strides=strides).reshape((C * HH * WW, N * Ho * Wo))

    # (F, C * HH * WW) @ (C * HH * WW, N * Ho * Wo) -> (F, N * Ho * Wo) -> (F, N, Ho, Wo) -> (N, F, Ho, Wo)
    result = Tensor((weight.data.reshape((F, -1)) @ x_strided).reshape((F, N, Ho, Wo)).swapaxes(1, 0), name="conv2d")
    
    if (weight.requires_grad or input.requires_grad) and weight.grad_enabled:
        def grad_fn(grad):
            # (N, F, Ho, Wo) -> (F, N, Ho, Wo) -> (F, N * Ho * Wo)
            grad_reshaped = grad.swapaxes(1, 0).reshape((F, -1))
            # (F, N * Ho * Wo) @ (N * Ho * Wo, C * HH * WW) -> (F, C * HH * WW) -> (F, C, HH, WW)
            weight_grad = (grad_reshaped @ x_strided.T).reshape(weight.data.shape)

            input_grad = None
            if input.requires_grad:
                # (C * HH * WW, F) @ (F, N * Ho * Wo) -> (C * HH * WW, N * Ho * Wo)
                x_strided_grad = weight.data.reshape((F, -1)).T @ grad_reshaped
                # (N * C * H * W, ) -> (C, HH, WW, N, Ho, Wo)
                x_idxs = TensorBase.as_strided(TensorBase.arange(N * C * H * W, device=input.device), shape=shape, strides=strides)
                # (N * C * H * W, )
                input_grad = TensorBase.zeros(N * C * H * W, device=input.device)
                input_grad.add_at(x_idxs.reshape(-1), x_strided_grad.reshape(-1))
                # (N * C * H * W, ) -> (N, C, H, W)
                input_grad = input_grad.reshape((N, C, H, W))

                if padding[0] or padding[1]:
                    input_grad = input_grad[..., padding[0] : -padding[0], padding[1] : -padding[1]]

            return (weight_grad, input_grad)

        result.grad_fn = Node(grad_fn=grad_fn,
                              next_functions=(weight.grad_fn, input.grad_fn),
                              name="conv2d")
        result.requires_grad = True

    if bias is not None:
        return result + bias.reshape((1, -1, 1, 1))

    return result


def avg_pool2d(input, kernel_size, stride=None, padding=0):
    N, C, H, W = input.shape
    HH, WW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    if stride is None:
        stride = (HH, WW)
    elif isinstance(stride, int):
        stride = (stride, stride)
    padding = (padding, padding) if isinstance(padding, int) else padding

    if padding[0] or padding[1]:
        H += 2 * padding[0]
        W += 2 * padding[1]
        x = TensorBase.zeros((N, C, H, W), device=input.device)
        x[..., padding[0] : -padding[0], padding[1] : -padding[1]] = input.data
    else:
        x = input.data

    Ho = (H - HH) // stride[0] + 1
    Wo = (W - WW) // stride[1] + 1

    shape = (N, C, Ho, Wo, HH, WW)
    strides = (C * H * W, H * W, stride[0] * W, stride[1], W, 1)

    # (N, C, H, W) -> (N, C, Ho, Wo, HH, WW) -> (N * C * Ho * Wo, HH * WW)
    x_strided = TensorBase.as_strided(x, shape=shape, strides=strides).reshape((N * C * Ho * Wo, HH * WW))
    
    # (N * C * Ho * Wo, HH * WW) -> (N * C * Ho * Wo, ) -> (N, C, Ho, Wo)
    result = Tensor((x_strided.sum(axis=-1) / (HH * WW)).reshape((N, C, Ho, Wo)), name="avg_pool2d")

    if input.requires_grad and input.grad_enabled:
        def grad_fn(grad):
            # (N, C, Ho, Wo) -> (N, C, Ho, Wo, HH, WW)
            grad_strided = TensorBase.as_strided(grad, shape=shape, strides=(C * Ho * Wo, Ho * Wo, Wo, 1, 0, 0))
            # (N * C * H * W, ) -> (N, C, Ho, Wo, HH, WW)
            x_idxs = TensorBase.as_strided(TensorBase.arange(N * C * H * W, device=input.device), shape=shape, strides=strides)
            # (N * C * H * W, )
            input_grad = TensorBase.zeros(N * C * H * W, device=input.device)
            input_grad.add_at(x_idxs.reshape(-1), grad_strided.reshape(-1) / (HH * WW))
            # (N * C * H * W, ) -> (N, C, H, W)
            input_grad = input_grad.reshape((N, C,  H,  W))

            if padding[0] or padding[1]:
                input_grad = input_grad[..., padding[0] : -padding[0], padding[1] : -padding[1]]

            return (input_grad, )

        result.grad_fn = Node(grad_fn=grad_fn,
                              next_functions=(input.grad_fn, ),
                              name="avg_pool2d")
        result.requires_grad = True

    return result


def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1):
    N, C, H, W = input.shape
    (HH, WW) = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
    if stride is None:
        stride = (HH, WW)
    elif isinstance(stride, int):
        stride = (stride, stride)
    padding = (padding, padding) if isinstance(padding, int) else padding

    if padding[0] or padding[1]:
        H += 2 * padding[0]
        W += 2 * padding[1]
        x = TensorBase.empty((N, C, H, W), device=input.device)
        x[:] = -inf
        x[..., padding[0] : -padding[0], padding[1] : -padding[1]] = input.data
    else:
        x = input.data

    Ho = (H - dilation[0] * (HH - 1) - 1) // stride[0] + 1
    Wo = (W - dilation[1] * (WW - 1) - 1) // stride[1] + 1

    shape = (N, C, Ho, Wo, HH, WW)
    strides = (C * H * W, H * W, stride[0] * W, stride[1], dilation[0] * W, dilation[1])

    # (N, C, H, W) -> (N, C, Ho, Wo, HH, WW) -> (N * C * Ho * Wo, HH * WW)
    x_strided = TensorBase.as_strided(x, shape=shape, strides=strides).reshape((N * C * Ho * Wo, HH * WW))

    # (N * C * Ho * Wo, HH * WW) -> (N * C * Ho * Wo, ) -> (N, C, Ho, Wo)
    result = Tensor(x_strided.max(axis=-1).reshape((N, C, Ho, Wo)), name="max_pool2d")

    if input.requires_grad and input.grad_enabled:
        def grad_fn(grad):
            # (N, C, Ho, Wo) -> (N, C, Ho, Wo, HH, WW)
            grad_strided = TensorBase.as_strided(grad, shape=shape, strides=(C * Ho * Wo, Ho * Wo, Wo, 1, 0, 0))
            # (N * C * H * W, ) -> (N, C, Ho, Wo, HH, WW)
            x_idxs = TensorBase.as_strided(TensorBase.arange(N * C * H * W, device=input.device), shape=shape, strides=strides)

            # (N * C * Ho * Wo, HH * WW)
            max_mask = TensorBase.arange(HH * WW, device=input.device) == x_strided.argmax(axis=-1, keepdims=True)
            
            # (N * C * H * W, )
            input_grad = TensorBase.zeros(N * C * H * W, device=input.device)
            input_grad.add_at(x_idxs.reshape(-1), grad_strided.reshape(-1) * max_mask.reshape(-1))
            # (N * C * H * W, ) -> (N, C, H, W)
            input_grad = input_grad.reshape((N, C, H, W))

            if padding[0] or padding[1]:
                input_grad = input_grad[..., padding[0] : -padding[0], padding[1] : -padding[1]]

            return (input_grad, )

        result.grad_fn = Node(grad_fn=grad_fn,
                              next_functions=(input.grad_fn, ),
                              name="max_pool2d")
        result.requires_grad = True

    return result
