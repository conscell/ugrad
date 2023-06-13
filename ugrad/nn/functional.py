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


def binary_cross_entropy(input, target):
    return -(target * input.log() + (1 - target) * (1 - input).log()).sum() / target.shape[0]


def nll_loss(input, target):
    return -(input * target).sum() / target.shape[0]


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