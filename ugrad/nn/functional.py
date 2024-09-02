from ..init import *
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


def softmax(input, dim=-1):
    return input.softmax(dim=dim)


def nll_loss(input, target):
    return -(input * target).sum() / target.shape[0]


def binary_cross_entropy(input, target):
    return -(target * input.log() + (1 - target) * (1 - input).log()).sum() / target.shape[0]


def cross_entropy(input, target):
    return nll_loss(log_softmax(input, dim=-1), Tensor(np.eye(input.shape[-1])[target]))


def dropout(input, p=0.5, training=True):
    if not training:
        return input
    drop_mask = np.random.uniform(size=input.shape) >= p
    result = Tensor((input.data * drop_mask) / (1 - p), name="dropout")

    if input.requires_grad and input.grad_enabled:
        # Define the gradient function for dropout
        result.grad_fn = Node(grad_fn=lambda grad: (drop_mask * grad / (1 - p), ),
                              next_functions=(input.grad_fn, ),
                              result_size=result.shape,
                              name="dropout")
        result.requires_grad = True

    return result


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05):
    norm_dim = ()
    h = 1.
    for nd, ns in enumerate(reversed(normalized_shape),1):
        if input.shape[-nd] != ns:
            raise RuntimeError("Incorrect input shape")
        norm_dim += (-nd, )
        h *= ns
    weight = weight if weight is not None else 1
    bias = bias if bias is not None else 0
    inp_zero_mean = input - input.sum(dim=norm_dim, keepdim=True) / h
    var = (inp_zero_mean ** 2).sum(dim=norm_dim, keepdim=True) / h
    result = inp_zero_mean / (var + eps) ** 0.5 * weight  + bias
    return result


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1):
    N = input.shape[0]
    H_in, W_in = input.shape[-2:]
    C_in = input.shape[1]
    C_out = weight.shape[0]
    kernel_size = weight.shape[-2:]
    dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
    dilated_size = (dilation[0] * (kernel_size[0] - 1) + 1, dilation[1] * (kernel_size[1] - 1) + 1)
    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding
    
    H_out = int((H_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    W_out = int((W_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)

    if padding[0] or padding[1]:
        x = np.zeros((N, C_in, H_in + 2 * padding[0], W_in + 2 * padding[1]))
        x[...,padding[0] : H_in + padding[0], padding[1] : W_in + padding[1]] = input.data
    else:
        x = input.data

    # Naïve implementation
    #
    # result = Tensor(np.zeros((N, C_out, H_out, W_out)), name="conv2d")
    #
    # for i in range(H_out):
    #     for j in range(W_out):
    #         result.data[...,i, j] = np.sum(weight.data * 
    #                                        np.expand_dims(x[...,i * stride[0] : i * stride[0] + dilated_size[0] : dilation[0],
    #                                                             j * stride[1] : j * stride[1] + dilated_size[1] : dilation[1]], axis=-4),
    #                                 axis=(-3, -2, -1))

    x_strided = np.lib.stride_tricks.sliding_window_view(
            x,
            dilated_size,
            axis=(-2, -1)
        )[..., ::stride[0] ,::stride[1], ::dilation[0], ::dilation[1]]
    
    result = Tensor(np.einsum("oikl, nihwkl -> nohw", weight.data, x_strided), name="conv2d")
    
    if weight.requires_grad and weight.grad_enabled:
        def grad_fn(grad):
            
            # Naïve implementation
            #
            # weight_grad = np.zeros_like(weight.data)
            #
            # for i in range(H_out):
            #     for j in range(W_out):
            #         weight_grad += np.sum(np.expand_dims(grad[...,i, j], axis=(-3, -2, -1)) * 
            #                               np.expand_dims(x[...,i * stride[0] : i * stride[0] + dilated_size[0] : dilation[0], 
            #                                                    j * stride[1] : j * stride[1] + dilated_size[1] : dilation[1]], axis=-4), 
            #                         axis=0)

            weight_grad = np.einsum("nohw, nihwkl -> oikl", grad, x_strided) if weight.requires_grad else None

            return (weight_grad, )

        result.grad_fn = Node(grad_fn=grad_fn,
                                next_functions=(weight.grad_fn, ),
                                result_size = result.shape,
                                name="conv2d")
        result.requires_grad = True

    if bias is None:
        return result

    return result + bias.reshape((-1, 1, 1))


def avg_pool2d(input, kernel_size, stride=None, padding=0):
    N = input.shape[0]
    H_in, W_in = input.shape[-2:]
    C = input.shape[1]
    kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    if stride is None:
        stride = kernel_size
    elif isinstance(stride, int):
        stride = (stride, stride)
    padding = (padding, padding) if isinstance(padding, int) else padding


    H_out = int((H_in + 2 * padding[0] - kernel_size[0]) / stride[0] + 1)
    W_out = int((W_in + 2 * padding[1] - kernel_size[1]) / stride[1] + 1)

    if padding[0] or padding[1]:
        x = np.zeros((N, C, H_in + 2 * padding[0], W_in + 2 * padding[1]))
        x[...,padding[0] : H_in + padding[0], padding[1] : W_in + padding[1]] = input.data
    else:
        x = input.data

    # Naïve implementation
    #
    # result = Tensor(np.zeros((N, C, H_out, W_out)), name="avg_pool2d")

    # for i in range(H_out):
    #     for j in range(W_out):
    #         result.data[...,i, j] = np.average(x[...,i * stride[0] : i * stride[0] + kernel_size[0],
    #                                                 j * stride[1] : j * stride[1] + kernel_size[1]], axis=(-2, -1))

    result = Tensor(np.average(
        np.lib.stride_tricks.sliding_window_view(x, kernel_size, axis=(-2, -1))[..., ::stride[0] ,::stride[1], :, :],
        axis=(-2, -1)
        ), name="avg_pool2d")
    
    if input.requires_grad and input.grad_enabled:
        def grad_fn(grad):
            kernel_numel = kernel_size[0] * kernel_size[1]
            
            # Naïve implementation
            #
            # input_grad = np.zeros_like(x)
            #
            # for i in range(H_out):
            #     for j in range(W_out):
            #         input_grad[...,i * stride[0] : i * stride[0] + kernel_size[0],
            #                        j * stride[1] : j * stride[1] + kernel_size[1]] += np.expand_dims(grad[...,i, j], 
            #                                                                                          axis=(-2, -1)) / kernel_numel

            input_grad = np.zeros(grad.shape[:-2] + ((grad.shape[-2] - 1) * stride[0] + 2 * kernel_size[0] - 1, 
                                                     (grad.shape[-1] - 1) * stride[1] + 2 * kernel_size[1] - 1))
            
            input_grad[..., kernel_size[0] - 1 : -kernel_size[0] + 1 : stride[0], 
                            kernel_size[1] - 1 : -kernel_size[1] + 1 : stride[1]] = grad
            
            input_grad = np.sum(np.lib.stride_tricks.sliding_window_view(input_grad, kernel_size, axis=(-2, -1)), axis=(-2, -1)) / kernel_numel
            
            input_grad = np.pad(input_grad, pad_width=((0, 0), 
                                                       (0, 0), 
                                                       (0, max(x.shape[-2] - input_grad.shape[-2], 0)),
                                                       (0, max(x.shape[-1] - input_grad.shape[-1], 0))))
            if padding:
                return (input_grad[...,padding[0] : H_in + padding[0], padding[1] : W_in + padding[1]], )
            
            return (input_grad, )

        result.grad_fn = Node(grad_fn=grad_fn,
                                next_functions=(input.grad_fn, ),
                                result_size = result.shape,
                                name="avg_pool2d")
        result.requires_grad = True

    return result