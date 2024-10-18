from .init import *


class Tensor:
    grad_enabled = True

    def __init__(self, data, name="", requires_grad=False):
        """
        A class representing a tensor object.
        This class provides functionality for tensor operations and gradient computations.

        Args:
            data: The data array or value.
            name: The name of the tensor (optional).
            requires_grad: Whether to compute gradients for this tensor (default: False).
        """
        self.data = data if isinstance(data, np.ndarray) else np.array(data)
        self.shape = self.data.shape
        self.name = name
        self.requires_grad = requires_grad
        self.retains_grad = False
        self.grad, self.grad_fn = (
            (np.zeros_like(self.data), Node(grad_fn=self.accum_grad, 
                                            next_functions=(),
                                            variable = self,
                                            name="accum")) if requires_grad and self.grad_enabled 
            else (None, None))
    
    def accum_grad(self, grad):
        """ 
        Gradient accumulation function.
        """
        self.grad += grad
        return ()
    
    @staticmethod
    def sum_to_size(x, size):
        """
        Sum the input ndarray `x` to the `size`. `size` must be expandable to the size of `x`.

        Args:
            x: Input ndarray.
            size: Desired shape of the output ndarray.

        Returns:
            The summed result of the input ndarray `x` adjusted to the `size`.

        """
        x_size = x.shape
        target_ndim = len(size)
        if target_ndim > x.ndim:
            raise RuntimeError(f"size {size} is not expandable to size {x_size}")
        if target_ndim < x.ndim:
            pre_axes = range(x.ndim - target_ndim)
            x = x.sum(tuple(pre_axes))
        axes = []
        for i, sz in enumerate(size):
            if sz != x.shape[i]:
                if sz == 1:
                    axes.append(i)
                else:
                    raise RuntimeError(f"size {size} is not expandable to size {x_size}")
        if axes:
            return x.sum(tuple(axes), keepdims=True)
        return x

    @property
    def T(self):
        """
        Return the transpose of the Tensor.

        Returns:
            The transposed Tensor object.
        """
        return self.permute()

    def permute(self, dims=None):
        """
        Permute the dimensions of the Tensor or return its transpose.

        Args:
            dims: The desired order of axes. If None, the Tensor is simply transposed (reverses all axes).

        Returns:
            A Tensor object with permuted dimensions based on the specified axes, or the transposed Tensor if no axes provided.
        """
        result = Tensor(np.transpose(self.data, axes=dims), name="T")

        if self.requires_grad and self.grad_enabled:
            # Define the gradient function for the transpose operation
            result.grad_fn = Node(grad_fn=lambda grad: (np.transpose(grad, axes=None if dims is None else sorted(range(len(dims)), key=dims.__getitem__)), ),
                                  next_functions=(self.grad_fn, ),
                                  variable=result,
                                  name="T")
            result.requires_grad = True
        
        return result

    def reshape(self, shape):
        """
        Reshape the Tensor to the specified shape.

        Args:
            shape (tuple or int): The desired shape of the output Tensor.

        Returns:
            A Tensor object with the specified shape.
        """
        result = Tensor(self.data.reshape(shape), name="reshape")

        if self.requires_grad and self.grad_enabled:
            # Define the gradient function for the reshape operation
            result.grad_fn = Node(grad_fn=lambda grad: (grad.reshape(self.shape), ),
                                  next_functions=(self.grad_fn, ),
                                  variable=result,
                                  name="reshape")
            result.requires_grad = True

        return result

    def __add__(self, other):
        """
        Add two Tensor objects element-wise.

        Args:
            other: The Tensor object or constant value to be added.

        Returns:
            The resulting Tensor object after the addition.
        """
        other_data, other_requires_grad, other_grad_fn = (
            (other.data, other.requires_grad, other.grad_fn) if isinstance(other, Tensor)
            else (other, False, None))

        result = Tensor(self.data + other_data, name="+")
        
        if (self.requires_grad or other_requires_grad) and self.grad_enabled:
            # Gradient function for element-wise addition of tensors with same shape
            def grad_fn(grad): return (
                grad if self.requires_grad else None, 
                grad if other_requires_grad else None)

            result.grad_fn = Node(grad_fn=grad_fn,
                                  next_functions=(self.grad_fn, other_grad_fn),
                                  variable=result,
                                  name="+")
            result.requires_grad = True

        return result

    def __mul__(self, other):
        """
        Multiply two Tensor objects element-wise.

        Args:
            other: The Tensor object or constant value to be multiplied.

        Returns:
            The resulting Tensor object after the multiplication.
        """
        other_data, other_requires_grad, other_grad_fn = (
            (other.data, other.requires_grad, other.grad_fn) if isinstance(other, Tensor)
            else (other, False, None))
        
        result = Tensor(self.data * other_data, name="*")

        if (self.requires_grad or other_requires_grad) and self.grad_enabled:
            # Gradient function for element-wise multiplication of tensors with same shape
            def grad_fn(grad): return (
                other_data * grad if self.requires_grad else None, 
                self.data * grad if other_requires_grad else None)
                
            result.grad_fn = Node(grad_fn=grad_fn,
                                  next_functions=(self.grad_fn, other_grad_fn),
                                  variable=result,
                                  name="*")
            result.requires_grad = True

        return result
    
    def __matmul__(self, other):
        """
        Perform matrix multiplication between two Tensor objects.

        Args:
            other: The Tensor object or constant value to be multiplied.

        Returns:
            The resulting Tensor object after the matrix multiplication.
        """
        other_data, other_requires_grad, other_grad_fn = (
            (other.data, other.requires_grad, other.grad_fn) if isinstance(other, Tensor)
            else (other, False, None))
        
        result = Tensor(self.data @ other_data, name="@")

        if (self.requires_grad or other_requires_grad) and self.grad_enabled:
            if self.data.ndim == other_data.ndim == 2:
                # Gradient function for matrix multiplication of 2D tensors
                def grad_fn(grad): return (
                    grad @ other_data.T if self.requires_grad else None, 
                    self.data.T @ grad if other_requires_grad else None)
            else:
                # Handling broadcasting for self when it is 1D
                self_expand_axis = (0, ) if self.data.ndim == 1 else ()
                
                # Handling broadcasting for other when it is 1D
                other_expand_axis = (-1, ) if other_data.ndim == 1 else ()
                
                # Determine the axes for broadcasting and reduction
                result_expand_axis = self_expand_axis + other_expand_axis

                # Gradient function for matrix multiplication
                def grad_fn(grad): return (
                    np.squeeze(np.expand_dims(grad, axis=result_expand_axis) @ 
                                                 np.expand_dims(other_data, axis=other_expand_axis).swapaxes(-1, -2),
                                                 axis=self_expand_axis) if self.requires_grad else None, 
                    np.squeeze(np.expand_dims(self.data, axis=self_expand_axis).swapaxes(-1, -2) @ 
                                                 np.expand_dims(grad, axis=result_expand_axis),
                                                 axis=other_expand_axis) if other_requires_grad else None)
                    
            result.grad_fn = Node(grad_fn=grad_fn,
                                  next_functions=(self.grad_fn, other_grad_fn),
                                  variable=result,
                                  name="@")
            result.requires_grad = True

        return result

    def __pow__(self, other):
        """
        Raise elements of Tensor object to the power of another element-wise.
        Note: The exponent 'other' is considered as a constant and not a variable, so no gradient is computed with respect to 'other'.

        Args:
            other: The exponent value or Tensor object containing the exponents.

        Returns:
            The resulting Tensor object contains bases in self raised to the exponents in other. 
        """
        other_data = other.data if isinstance(other, Tensor) else other

        result = Tensor(self.data ** other_data, name="**")

        if self.requires_grad and self.grad_enabled:
            # Define the gradient function for ** operation
            result.grad_fn = Node(grad_fn=lambda grad: (other_data * self.data ** (other_data - 1) * grad, ),
                                  next_functions=(self.grad_fn, ),
                                  variable=result,
                                  name="**")
            result.requires_grad = True

        return result
    
    def sum(self, dim=None, keepdim=False):
        """
        Compute the sum of elements in the Tensor.

        Args:
            dim: The dimension or dimensions to reduce. If None, all dimensions are reduced (optional).
            keepdim: If True the axes which are reduced are left in the result as dimensions with size 1 (default: False).
        Returns:
            The resulting Tensor object representing the sum.
        """
        result = Tensor(np.sum(self.data, axis=dim, keepdims=keepdim), name="sum")

        if self.requires_grad and self.grad_enabled:
            expand_axis = dim if dim and not keepdim else ()
            # Define the gradient function for summation
            result.grad_fn = Node(grad_fn=lambda grad: (np.ones_like(self.data) * np.expand_dims(grad, axis=expand_axis), ),
                                  next_functions=(self.grad_fn, ),
                                  variable=result,
                                  name="sum")
            result.requires_grad = True
        
        return result

    def exp(self):
        """
        Compute the exponential of each element in the Tensor.

        Returns:
            The resulting Tensor object representing the exponential.
        """
        result = Tensor(np.exp(self.data), name="exp")

        if self.requires_grad and self.grad_enabled:
            # Define the gradient function for exponent
            result.grad_fn = Node(grad_fn=lambda grad: (result.data * grad, ),
                                  next_functions=(self.grad_fn, ),
                                  variable=result,
                                  name="exp")
            result.requires_grad = True

        return result
    
    def log(self):
        """
        Compute the natural logarithm of each element in the Tensor.

        Returns:
            The resulting Tensor object representing the logarithm.
        """
        result = Tensor(np.log(self.data), name="log")

        if self.requires_grad and self.grad_enabled:
            # Define the gradient function for logarithm
            result.grad_fn = Node(grad_fn=lambda grad: (grad / self.data, ),
                                  next_functions=(self.grad_fn, ),
                                  variable=result,
                                  name="log")
            result.requires_grad = True

        return result
    
    def relu(self):
        """
        Apply the Rectified Linear Unit (ReLU) activation function element-wise.

        Returns:
            The resulting Tensor object after applying ReLU.
        """
        result = Tensor(np.maximum(0, self.data), name="relu")
        
        if self.requires_grad and self.grad_enabled:
            # Define the gradient function for ReLU
            result.grad_fn = Node(grad_fn=lambda grad: ((self.data > 0) * grad, ),
                                  next_functions=(self.grad_fn, ),
                                  variable=result,
                                  name="relu")
            result.requires_grad = True

        return result

    def sigmoid(self):
        """
        Apply the sigmoid activation function element-wise.

        Returns:
            The resulting Tensor object after applying the sigmoid function.
        """
        result = Tensor(np.tanh(self.data * 0.5) * 0.5 + 0.5, name="sigmoid")
        
        if self.requires_grad and self.grad_enabled:
            # Define the gradient function for sigmoid
            result.grad_fn = Node(grad_fn=lambda grad: (result.data * (1 - result.data) * grad, ),
                                  next_functions=(self.grad_fn, ),
                                  variable=result,
                                  name="sigmoid")
            result.requires_grad = True

        return result

    def tanh(self):
        """
        Apply the hyperbolic tangent (tanh) activation function element-wise.

        Returns:
            The resulting Tensor object after applying the tanh function.
        """
        result = Tensor(np.tanh(self.data), name="tanh")
        
        if self.requires_grad and self.grad_enabled:
            # Define the gradient function for tanh
            result.grad_fn = Node(grad_fn=lambda grad: ((1 - result.data**2) * grad, ),
                                  next_functions=(self.grad_fn, ),
                                  variable=result,
                                  name="tanh")
            result.requires_grad = True

        return result
    
    def log_softmax(self, dim=-1):
        """
        Applies a softmax function followed by a logarithm.

        Args:
            dim: The dimension along which to compute the log-softmax (default: -1).

        Returns:
            The resulting Tensor object after applying the log-softmax function.
        """
        logits_off = self - np.max(self.data, axis=dim, keepdims=True)
        result = logits_off - logits_off.exp().sum(dim=dim, keepdim=True).log()
        result.name = "log_softmax"
        return result
    
    def softmax(self, dim=-1):
        """
        Applies a softmax function.

        Args:
            dim: The dimension along which to compute the softmax (default: -1).

        Returns:
            The resulting Tensor object after applying the softmax function.
        """
        logits_off = self - np.max(self.data, axis=dim, keepdims=True)
        logits_off_exp = logits_off.exp()
        result = logits_off_exp / logits_off_exp.sum(dim=dim, keepdim=True)
        result.name = "softmax"
        return result

    # Other operations
    def __neg__(self):
        return self * (-1)

    def __sub__(self, other):
        return self + (-other)
    
    def __radd__(self, other):
        return self + other
    
    def __rsub__(self, other):
        return (-self) + other

    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * other ** (-1)

    def __rtruediv__(self, other):
        return self ** (-1) * other

    def __repr__(self):
        """
        Return a string representation of the Tensor object.

        Returns:
            The string representation of the Tensor object.
        """
        return f"Tensor(data={self.data}, grad={self.grad}, name={self.name} requires_grad={self.requires_grad})"

    def backward(self, gradient=None, retain_graph=False):
        """
        Perform backpropagation to compute gradients for all Node objects involved in the computation graph
        and store the gradients in the leaf Tensor nodes requiring gradient computation.

        This method traverses the computation graph, starting from the Node object associated with the current 
        Tensor object and propagating gradients to the previous Node objects using their respective gradient functions.

        Args:
            gradient: Gradient w.r.t. the tensor. 
            retain_graph: If True, the graph used to compute the grads will be retained, otherwise it will be freed (default: False).

        Returns:
            None
        """
        topo = set()
        stack = [self.grad_fn]
        while stack:
            grad_fn = stack.pop()
            if grad_fn not in topo:
                for fn in grad_fn.next_functions:
                    if fn:
                        stack.append(fn)
                grad_fn.grad = np.zeros_like(grad_fn.variable.data)
                topo.add(grad_fn)

        topo = sorted(topo, key=lambda n: n.topological_nr, reverse=True)

        self.grad_fn.grad = gradient if gradient is not None else np.ones_like(self.data)        
        for grad_fn in topo:
            for fn, gr in zip(grad_fn.next_functions, grad_fn(grad_fn.grad)):
                if fn:
                    fn.grad += gr if gr.shape == fn.variable.shape else self.sum_to_size(gr, fn.variable.shape)

            if grad_fn.variable.retains_grad:
                if grad_fn.variable.grad is not None:
                    grad_fn.variable.grad += grad_fn.grad
                else:
                    grad_fn.variable.grad = grad_fn.grad

            if grad_fn.name != "accum" and not retain_graph:
                grad_fn.grad_fn = None
                grad_fn.variable = None

            grad_fn.grad = None


class Node:
    def __init__(self, grad_fn, next_functions, variable, name=""):
        """
        A class representing a gradient function node in the computational graph.
        Gradient function nodes encapsulate the gradient computation and propagation
        for a specific operation in the graph.

        Args:
            grad_fn: The gradient function.
            next_functions: A tuple of next gradient function nodes.
            variable: The variable associated with this node in the computation.
            name: The name of the gradient function node (optional).
        """
        self.grad_fn = grad_fn
        self.next_functions = next_functions
        self.variable = variable
        self.name = name
        self.topological_nr = 0
        for fn in next_functions:
            if fn and self.topological_nr <= fn.topological_nr:
                self.topological_nr = fn.topological_nr + 1

    def __call__(self, grad):
        """
        Call the gradient function with the given gradient.

        Args:
            grad: The gradient to be passed to the gradient function.

        Returns:
            The result of the gradient function.
        """
        if self.grad_fn:
            return self.grad_fn(grad)
        else:
            raise RuntimeError("Trying to backward through the graph a second time.")
    
    def __repr__(self):
        """
        Return a string representation of the gradient function node.

        Returns:
            A string representation of the gradient function node.
        """
        return f"Node={self.name}"


class no_grad:
    def __init__(self):
        """
        Initialize the 'no_grad' context manager.
        """
        self.prev = False
    
    def __enter__(self):
        """
        Disable gradient computation.
        
        This method is called when entering a 'with' block.
        It temporarily disables the computation of gradients by setting the 'grad_enabled' flag to False.
        """
        self.prev = Tensor.grad_enabled
        Tensor.grad_enabled = False

    def __exit__(self, *args):
        """
        Enable gradient computation.
        
        This method is called when exiting a 'with' block.
        It restores the previous value of the 'grad_enabled' flag, allowing gradient computation to resume.
        """
        Tensor.grad_enabled = self.prev
