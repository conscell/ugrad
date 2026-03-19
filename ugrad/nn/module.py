from ..tensor import Tensor
from ..tensorbase import TensorBase
from .functional import conv2d, avg_pool2d, max_pool2d

class Module:
    def __init__(self):
        """
        Base class for neural network modules.
        """
        self._parameters = {}
        self._modules = {}

    def __setattr__(self, name, value):
        """
        Overrides the default attribute setting behavior.

        This method is called when an attribute is set on an instance of the Module class.
        It handles special cases for parameters and sub-modules.
        """
        if isinstance(value, Tensor):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        """
        Overrides the default attribute getting behavior.

        This method is called when an attribute is accessed on an instance of the Module class.
        It handles special cases for parameters and sub-modules.
        """
        if name in self._parameters:
            return self._parameters[name]
        elif name in self._modules:
            return self._modules[name]
        else:
            return super().__getattr__(name)

    def __call__(self, *args, **kwargs):
        """
        Allows the Module instance to be called like a function.

        This method enables the Module instance to be called like a function
        by forwarding the call to the forward method. Any arguments passed to the call
        are forwarded to the forward method.

        Returns:
        - The result of the forward method.
        """
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        """
        Performs the forward pass through the module.

        This method performs the forward pass through the module.
        It should be implemented by subclasses to define the actual computation
        that takes place during the forward pass.
        """
        raise NotImplementedError

    def modules(self):
        """
        Returns an iterator over all sub-modules.

        This method allows access to all the sub-modules contained within the current module.
        It returns an iterator that can be used to iterate over the sub-modules.
        """
        return iter(self._modules.values())  # Return an iterator over the sub-modules

    def parameters(self):
        """
        Returns an iterator over all parameters in the module and its sub-modules.

        This method provides access to all the parameters present in the module and its sub-modules.
        It returns an iterator that can be used to iterate over the parameters.
        """
        for module in self._modules.values():
            yield from module.parameters()  # Recursively yield parameters from sub-modules
        yield from self._parameters.values()  # Yield parameters from the current module


class Linear(Module):
    def __init__(self, in_features, out_features, name=""):
        """
        Linear layer applies a linear transformation to the input: y = x W.T + b
        input x [*, in_features]
        learnable weights W [out_features, in_features]
        bias b [out_features]
        output y [*, out_features]

        Args:
            in_features: The number of input features.
            out_features: The number of output features.
            name: The name of the Linear layer (optional).
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.name = name

        # Initialize weight and bias with random values
        bound = self.in_features ** -0.5
        self.weight = Tensor(TensorBase.empty((out_features, in_features), dtype="double").uniform_(-bound, bound),
                             requires_grad=True, 
                             name="w_" + name)
        self.bias = Tensor(TensorBase.empty((out_features, ), dtype="double").uniform_(-bound, bound), 
                           requires_grad=True, 
                           name="b_" + name)

    def forward(self, inp):
        """
        Perform a forward pass through the Linear layer.

        Args:
            inp: The input tensor.

        Returns:
            The result of the forward pass.
        """
        return (inp @ self.weight.T + self.bias)



class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, name=""):
        """
        2D Convolutional layer applies a 2D convolution to the input.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of filters (output channels).
            kernel_size: Size of the convolutional filter.
            stride: Step size for moving the filter across the input.
            padding: Number of elements to pad around the input.
            dilation: Dilation factor for increasing the receptive field of the filter.
            bias: If True, a bias term is added to the output.
            name: The name of the Conv2d layer (optional).
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.name = name

        # Initialize weight and bias with random values
        bound = (self.in_channels * self.kernel_size[0] * self.kernel_size[1]) ** -0.5
        self.weight = Tensor(TensorBase.empty((out_channels, 
                                               in_channels, 
                                               self.kernel_size[0],
                                               self.kernel_size[1])).uniform_(-bound, bound),
                             requires_grad=True, 
                             name="w_" + name)
        if bias:
            self.bias = Tensor(TensorBase.empty((out_channels, )).uniform_(-bound, bound),
                            requires_grad=True, 
                            name="b_" + name)
        else:
            self.bias = None

    def forward(self, inp):
        """
        Perform a forward pass through the Conv2d layer.

        Args:
            inp: The input tensor.

        Returns:
            The result of the forward pass.
        """
        return conv2d(inp, self.weight, bias=self.bias, 
                      stride=self.stride, padding=self.padding, dilation=self.dilation)


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, name=""):
        """
        AvgPool2d layer applies a 2D average pooling to the input.

        Args:
            kernel_size: Size of the pooling window.
            stride: Step size for moving the pooling window across the input. 
            padding: Number of elements to pad around the input.
            name: The name of the AvgPool2d layer (optional).
        """
        super().__init__()
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.name = name

    def forward(self, inp):
        """
        Perform a forward pass through the AvgPool2d layer.

        Args:
            inp: The input tensor.

        Returns:
            The result of the forward pass.
        """
        return avg_pool2d(inp, self.kernel_size, stride=self.stride, padding=self.padding)
    

class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, name=""):
        """
        MaxPool2d layer applies a 2D maximum pooling to the input.

        Args:
            kernel_size: Size of the pooling window.
            stride: Step size for moving the pooling window across the input. 
            padding: Number of elements to pad around the input.
            name: The name of the MaxPool2d layer (optional).
        """
        super().__init__()
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.name = name

    def forward(self, inp):
        """
        Perform a forward pass through the MaxPool2d layer.

        Args:
            inp: The input tensor.

        Returns:
            The result of the forward pass.
        """
        return max_pool2d(inp, self.kernel_size, stride=self.stride, padding=self.padding)