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
