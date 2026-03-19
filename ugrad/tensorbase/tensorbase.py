import ctypes
import os
import numpy as np


class CStorage(ctypes.Structure):
    _fields_ = [
        ('nbytes', ctypes.c_int),   # Number of bytes allocated
        ('nshares', ctypes.c_int),  # Number of shared references
        ('device', ctypes.c_int),   # Device type (CPU = -1 / GPU = 0, 1, ...)
        ('data', ctypes.c_void_p)   # Pointer to raw memory
    ]
    def __repr__(self):
            return (f"CStorage(nbytes={self.nbytes}, nshares={self.nshares}, "
                    f"device={self.device}, data={self.data if self.data else None})")

class CTensor(ctypes.Structure):
    _fields_ = [
        ('storage', ctypes.POINTER(CStorage)),      # Pointer to storage
        ('shape', ctypes.POINTER(ctypes.c_int)),    # Pointer to shape array
        ('strides', ctypes.POINTER(ctypes.c_int)),  # Pointer to strides array
        ('offset', ctypes.c_int),                   # Storage offset
        ('ndim', ctypes.c_int),                     # Number of dimensions
        ('numel', ctypes.c_int),                    # Number of elements
        ('dtype', ctypes.c_int),                    # Data type (double = 0, long = 1)
        ('device', ctypes.c_int),                   # Device type (CPU = -1 / GPU = 0, 1, ...)
    ]

    def __repr__(self):
        shape_str = f"({', '.join(str(self.shape[i]) for i in range(self.ndim))})"
        strides_str = f"({', '.join(str(self.strides[i]) for i in range(self.ndim))})"
        return (f"CTensor(storage={self.storage.contents if self.storage else None}, shape={shape_str}, strides={strides_str}, "
                f"offset={self.offset}, ndim={self.ndim}, numel={self.numel}, dtype={self.dtype}, device={self.device})")

module_dir = os.path.dirname(os.path.abspath(__file__))
_C = ctypes.CDLL(os.path.join(module_dir, "libtensor.so"))
_C.create_tensor.argtypes = [ctypes.POINTER(CStorage), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
_C.create_tensor.restype = ctypes.POINTER(CTensor)
_C.create_storage.argtypes = [ctypes.c_int, ctypes.c_int]
_C.create_storage.restype = ctypes.POINTER(CStorage)
_C.cc_storage.argtypes = [ctypes.c_int, ctypes.c_void_p]
_C.cc_storage.restype = ctypes.POINTER(CStorage)
_C.delete_tensor.argtypes = [ctypes.POINTER(CTensor)]
_C.delete_tensor.restype = None
_C.update_tensor.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
_C.update_tensor.restype = None
_C.clone_tensor.argtypes = [ctypes.POINTER(CTensor)]
_C.clone_tensor.restype = ctypes.POINTER(CTensor)
_C.to.argtypes = [ctypes.POINTER(CTensor), ctypes.c_int]
_C.to.restype = ctypes.POINTER(CTensor)
_C.update_tensor.restype = None
_C.get_item.argtypes = [ctypes.POINTER(CTensor), ctypes.c_int]
_C.get_item.restype = ctypes.c_void_p
_C.is_contiguous.argtypes = [ctypes.POINTER(CTensor)]
_C.is_contiguous.restype = ctypes.c_bool
_C.sum.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
_C.sum.restype = ctypes.POINTER(CTensor)
_C.max_t.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
_C.max_t.restype = ctypes.POINTER(CTensor)
_C.argmax.argtypes = [ctypes.POINTER(CTensor), ctypes.c_int]
_C.argmax.restype = ctypes.POINTER(CTensor)
_C.gt.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
_C.gt.restype = ctypes.POINTER(CTensor)
_C.eq.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
_C.eq.restype = ctypes.POINTER(CTensor)
_C.add.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
_C.add.restype = ctypes.POINTER(CTensor)
_C.mul.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
_C.mul.restype = ctypes.POINTER(CTensor)
_C.assign.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
_C.assign.restype = None
_C.add_at.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
_C.add_at.restype = None
_C.uniform.argtypes = [ctypes.POINTER(CTensor), ctypes.c_double, ctypes.c_double]
_C.uniform.restype = None
_C.maximum.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
_C.maximum.restype = ctypes.POINTER(CTensor)
_C.mul_reduce.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int]
_C.mul_reduce.restype = ctypes.POINTER(CTensor)
_C.pow_t.argtypes = [ctypes.POINTER(CTensor), ctypes.c_double]
_C.pow_t.restype = ctypes.POINTER(CTensor)
_C.exp_t.argtypes = [ctypes.POINTER(CTensor)]
_C.exp_t.restype = ctypes.POINTER(CTensor)
_C.log_t.argtypes = [ctypes.POINTER(CTensor)]
_C.log_t.restype = ctypes.POINTER(CTensor)
_C.tanh_t.argtypes = [ctypes.POINTER(CTensor)]
_C.tanh_t.restype = ctypes.POINTER(CTensor)
_C.contiguous.argtypes = [ctypes.POINTER(CTensor)]
_C.contiguous.restype = ctypes.POINTER(CTensor)
_C.arange.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_C.arange.restype = ctypes.POINTER(CTensor)

dtypes = {"double": (0, ctypes.c_double), "long": (1, ctypes.c_long)}
code2dtype = {0: "double", 1: "long"}
npdtypes = {"float64": "double", "int64": "long"}
inf = float("inf")

def flatten(x, depth=1, shape=None):
    """
    Flattens a nested iterable and computes the shape of the original structure.

    Args:
        x (iterable): The nested iterable to be flattened.
        depth (int, optional): Internal parameter for tracking depth.
        shape (list, optional): Internal parameter to store the shape of the input structure.

    Returns:
        tuple: A tuple containing:
            - list: Flattened x.
            - list: The shape of the original nested structure.
    """
    result = []
    shape = [] if shape is None else shape
    if len(shape) < depth:
        shape.append(len(x))
    assert shape[depth - 1] == len(x) 
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, str):
            flat = flatten(el, depth=depth + 1, shape=shape)
            result.extend(flat[0])
        else:
            result.append(el)
    return result, shape

class TensorBase:
    """
    A base class for multi-dimensional tensors.

    Args:
        data (TensorBase | CTensor ptr | np.ndarray | list | int | float): The input data to initialize the tensor.
        shape (tuple, optional): Shape (used when creating from lists).
        dtype (str, optional): Data type ('double' or 'long'). Defaults to inferred type.
        device (str, optional): The device on which to allocate the tensor ('cpu' or 'cuda'). Defaults to 'cpu'.

    Raises:
        TypeError: If the input data type is not supported.
    """
    def __init__(self, data, shape=(), dtype=None, device="cpu"):
        if isinstance(data, TensorBase):
            # Initialize from an existing TensorBase instance
            self.dtype = data.dtype
            self.device = data.device
            self.shape = data.shape
            self.ndim = data.ndim

            # Create a new tensor reusing the existing tensor's storage
            self.data = _C.create_tensor(
                data.data.contents.storage,                 # storage
                (ctypes.c_int * self.ndim)(*self.shape),    # shape
                ctypes.c_int(self.ndim),                    # ndim
                ctypes.c_int(dtypes[self.dtype][0]),        # dtype code
            )

        elif isinstance(data, ctypes.POINTER(CTensor)):
            # Initialize from a CTensor ptr
            self.dtype = code2dtype[data.contents.dtype]
            self.device = "cpu" if data.contents.device == -1 else "cuda"
            self.shape = tuple(ctypes.cast(data.contents.shape, ctypes.POINTER(ctypes.c_int * data.contents.ndim)).contents)
            self.ndim = data.contents.ndim
            self.data = data  # Store the raw pointer

        elif isinstance(data, np.ndarray):
            # Convert from a NumPy array
            self.device = "cpu"
            self.shape = data.shape
            self.ndim = data.ndim
            if dtype is not None:
                self.dtype = dtype
                data = data.astype(dtype)
            else:
                self.dtype = npdtypes[str(data.dtype)]

            self.data = _C.create_tensor(
                _C.cc_storage(ctypes.c_int(data.itemsize * data.size),                        # nbytes
                              data.ctypes.data_as(ctypes.c_void_p)                            # data
                             ),                              # storage
                data.ctypes.shape_as(ctypes.c_int),          # shape
                ctypes.c_int(self.ndim),                     # ndim
                ctypes.c_int(dtypes[self.dtype][0]),         # dtype code
            )

        elif isinstance(data, list):
            # Convert from a nested list
            self.device = "cpu"
            data, shape = flatten(data)
            self.shape = tuple(shape)
            self.ndim = len(self.shape)
            if dtype == "double":
                data = list(map(float, data))  # Promote mixed int/float to float
                self.dtype = dtype
            elif dtype == "long":
                data = list(map(int, data))
                self.dtype = dtype
            else:
                types = set(map(type, data))
                if len(types) == 2 and float in types and int in types:
                    data = list(map(float, data))
                    self.dtype = "double"
                elif len(types) == 1 and float in types:
                    self.dtype = "double"
                elif len(types) == 1 and int in types:
                    self.dtype = "long"
                else:
                    raise TypeError(f"Invalid data type: {types}") 
            
            # Create tensor from list data
            self.data = _C.create_tensor(
                _C.cc_storage(ctypes.c_int(ctypes.sizeof(dtypes[self.dtype][1]) * len(data)), # nbytes
                              (dtypes[self.dtype][1] * len(data))(*data)                      # data
                             ),                              # storage
                (ctypes.c_int * self.ndim)(*self.shape),     # shape
                ctypes.c_int(self.ndim),                     # ndim
                ctypes.c_int(dtypes[self.dtype][0]),         # dtype code
            )

        elif isinstance(data, int) or isinstance(data, float):
            # Handle scalar values
            self.dtype = dtype if dtype is not None else ("double" if isinstance(data, float) else "long")
            self.device = "cpu"
            self.shape = ()
            self.ndim = 0

            # Create tensor for a single scalar
            self.data = _C.create_tensor(
                _C.cc_storage(ctypes.c_int(ctypes.sizeof(dtypes[self.dtype][1])),             # nbytes
                              (dtypes[self.dtype][1] * 1)(data)                               # data
                             ),                              # storage
                (ctypes.c_int * self.ndim)(*self.shape),     # shape
                ctypes.c_int(self.ndim),                     # ndim
                ctypes.c_int(dtypes[self.dtype][0]),         # dtype code
            )
        else:
            raise TypeError(f"Invalid data type: {type(data)}") 

        # Set tensor strides and number of elements
        self.strides = tuple(ctypes.cast(self.data.contents.strides, ctypes.POINTER(ctypes.c_int * self.ndim)).contents)
        self.numel = self.data.contents.numel
        
        # Move to CUDA if requested
        if device == "cuda":
            self.device = "cuda"
            self.data = _C.to(self.data, 0)

    def to(self, device):
        """
        Moves the tensor to the specified device.

        Args:
            device (str): Target device, either 'cpu' or 'cuda'.

        Returns:
            TensorBase: A new tensor on the specified device if different, otherwise self.
        """
        if self.device == device:
            return self
        return TensorBase(_C.to(self.data, -1 if device =="cpu" else 0))

    def clone(self):
        """
        Creates a deep copy of the tensor.

        Returns:
            TensorBase: A new tensor with the same data and properties.
        """
        return TensorBase(_C.clone_tensor(self.data))
    
    def __repr__(self):
        """
        Returns a string representation of the tensor, preserving its shape.

        Returns:
            str: A formatted string representation of the tensor.
        """
        # Compute logical strides for reconstructing the tensor from its flattened form
        logical_strides = []
        stride = 1
        for i in range(self.ndim - 1, -1, -1):
            logical_strides = [stride] + logical_strides
            stride *= self.shape[i]
        
        # Retrieve flattened data from the C backend
        flat_data = [dtypes[self.dtype][1].from_address(_C.get_item(self.data, i)).value for i in range(self.numel)]
        
        # Start building the representation string
        repr_str = "[" * self.ndim  # Open brackets for each dimension
        for i in range(len(flat_data)):
            brackets_start = ""
            brackets_end = ""
            for stride in logical_strides[:-1]:
                if i % stride == 0:
                    brackets_start += "["
                    brackets_end += "]"
            
            # Add appropriate spacing, commas, and line breaks
            if i > 0:
                if brackets_end:
                    repr_str += brackets_end + "," + "\n" * len(brackets_end)
                else:
                    repr_str += ", "
                
                if brackets_start:
                    repr_str += " " * (self.ndim - len(brackets_start)) + brackets_start

            repr_str += str(flat_data[i])

        repr_str += "]" * self.ndim  # Close all brackets at the end

        return repr_str

    def __getitem__(self, indices):
        """
        Retrieves a slice or a specific item from the tensor.

        Args:
            indices (tuple): A tuple of indices or slices. Can include integer indices, slices, or ellipsis.

        Returns:
            TensorBase: A new tensor containing the indexed or sliced values.
        """
        if not isinstance(indices, tuple):
            indices = (indices, )
        shape = []
        strides = []

         # Handle ellipsis which automatically fills in the missing dimensions
        if ... in indices:
            ellipsis_idx = indices.index(...)
            # Replace ellipsis with slices for all remaining dimensions
            indices = indices[:ellipsis_idx] + (slice(None), ) * (self.ndim - len(indices) + 1)  + indices[ellipsis_idx + 1:]
        
        offset = 0 # This the offset into the flattened data based on the indices
        for idx, dim, stride in zip(indices, self.shape, self.strides):
            if isinstance(idx, int):
                # For integer indices, calculate the offset directly
                offset += idx * stride
            else:
                # For slice indices, handle start, stop, and step values
                start = idx.start if idx.start is not None else 0
                stop = idx.stop if idx.stop is not None else dim
                
                if start < -dim:
                    start = 0
                elif start < 0:
                    start = dim + start
                elif start > dim:
                    start = dim

                if stop < -dim:
                    stop = 0
                elif stop < 0:
                    stop = dim + stop
                elif stop > dim:
                    stop = dim

                step = idx.step if idx.step is not None else 1

                offset += start * stride
                shape.append(-((stop - start) // -step))
                strides.append(stride * step)

        # Extend the shape and strides with the remaining dimensions of the original tensor
        shape.extend(self.shape[len(indices):])
        strides.extend(self.strides[len(indices):])
        
        # Create a new tensor with the sliced data
        result = TensorBase(self)
        result.ndim = len(shape)
        result.shape = tuple(shape)
        result.strides = tuple(strides)

        # Update the tensor's data with the new shape, strides, and number of dimensions
        _C.update_tensor(
            result.data,
            (ctypes.c_int * result.ndim)(*result.shape),
            (ctypes.c_int * result.ndim)(*result.strides),
            result.ndim
        )

        # Set the offset in the CTensor data structure
        result.data.contents.offset = offset
         
         # Update the number of elements in the tensor
        result.numel = result.data.contents.numel

        return result
    
    def __setitem__(self, indices, other):
        """
        Assigns values to the tensor at the specified indices.

        Args:
            indices (tuple): The indices of the elements to update. Can include integer indices, slices, or ellipsis.
            other (TensorBase | int | float): The value to assign.
        
        Raises:
            RuntimeError: If `other` cannot be broadcast to the shape of the indexed result.
        """
        if isinstance(other, float):
            other = TensorBase(other, device=self.device)
        elif isinstance(other, int):
            other = TensorBase(other, dtype="long", device=self.device)
        
        result = self[indices]
        # Broadcast if shapes are different
        if result.shape != other.shape:
            broadcast_shape = TensorBase.broadcast_shape(result.shape, other.shape)
            if result.shape != broadcast_shape:
                raise RuntimeError(f"Incorrect shape: {other.shape}")
            other = other.broadcast_to(broadcast_shape)
        _C.assign(result.data, other.data)

    def numpy(self):
        """
        Converts the tensor to a NumPy array.

        Returns:
            np.ndarray: A NumPy array with the same data as the tensor.

        Raises:
            TypeError: If the tensor is not stored on the CPU.
        """
        if self.device != "cpu":
            raise TypeError(f"TypeError: can't convert {self.device} device type tensor to numpy.")
        result = self.contiguous()
        return np.ctypeslib.as_array(ctypes.cast(result.data.contents.storage.contents.data,
                                                 ctypes.POINTER(dtypes[self.dtype][1])), shape=self.shape).astype(self.dtype)
    
    def reshape(self, shape):
        """
        Reshapes the tensor to the specified shape.

        Args:
            shape (tuple): The new shape for the tensor.

        Returns:
            TensorBase: A new tensor with the specified shape.

        Raises:
            RuntimeError: If the new shape is incompatible with the number of elements in the tensor.
        """
        # Ensure that the new shape is correct
        if isinstance(shape, int):
            shape = (shape, )
        numel = 1
        for dim in shape:
            if dim > 0 or (dim == -1 and numel > 0):
                numel *= dim
            else:
                raise RuntimeError(f"Incorrect shape: {shape}")       
        if numel > 0 and numel != self.numel:
            raise RuntimeError(f"Incorrect shape: {shape}")
        # If -1 is used, determine its correct value
        if numel < 0:
            if self.numel % -numel != 0:
                raise RuntimeError(f"Incorrect shape: {shape}")
            missing_dim = self.numel // -numel
            result_shape = [dim if dim !=-1 else missing_dim for dim in shape]
        else:
            result_shape = list(shape)

        # Ensure the tensor is contiguous before reshaping
        result = TensorBase(self) if self.is_contiguous() else TensorBase(_C.contiguous(self.data))
        
        # Update shape and number of dimensions
        result.shape = tuple(result_shape)
        result.ndim = len(result_shape)

        # Compute new strides for the reshaped tensor 
        strides = []
        stride = 1
        for i in range(result.ndim - 1, -1, -1):
            strides = [stride] + strides
            stride *= result.shape[i]
        result.strides = tuple(strides)

        # Update the tensor metadata in the C backend
        _C.update_tensor(
            result.data,
            (ctypes.c_int * result.ndim)(*result.shape),
            (ctypes.c_int * result.ndim)(*result.strides),
            result.ndim
        )
        return result
    
    def as_strided(self, shape, strides):
        """
        Returns a view of the tensor with the specified shape and strides.

        Args:
            shape: The desired shape of the returned tensor.
            strides: The desired strides of the returned tensor.

        Returns:
            TensorBase: A new tensor view sharing the same underlying data.

        Raises:
            AssertionError: If the number of dimensions in `shape` and `strides` do not match.
            RuntimeError: If any dimension in `shape` is less than or equal to 0.
        """
        result_ndim = len(shape)
        
        assert result_ndim == len(strides)

        for dim in shape:
            if dim <= 0:
                raise RuntimeError(f"Incorrect shape: {shape}")

        result = TensorBase(self)
        result.shape = shape
        result.strides = strides
        result.ndim = result_ndim

        # Update the tensor metadata in the C backend
        _C.update_tensor(
            result.data,
            (ctypes.c_int * result.ndim)(*result.shape),
            (ctypes.c_int * result.ndim)(*result.strides),
            result.ndim
        )
        result.numel = result.data.contents.numel
        return result

    def broadcast_to(self, shape):
        """
        Expands the tensor to the given shape using broadcasting rules.

        Args:
            shape (tuple): The target shape to broadcast the tensor to.

        Returns:
            TensorBase: A new tensor with the specified shape and adjusted strides.

        Raises:
            RuntimeError: If the tensor cannot be broadcast to the given shape.
        """
        strides = []
        result_ndim = len(shape)
        # Pad the original shape and strides with 1s and 0s to match the target dimensionality
        shape_padded = (1, ) * (result_ndim - self.ndim) + self.shape
        strides_padded = (0, ) * (result_ndim - self.ndim) + self.strides
        
        # Compute new strides based on broadcasting rules
        for i in range(result_ndim):
            if shape_padded[i] == 1:
                strides.append(0)  # Dimension is broadcasted; stride is set to 0
            elif shape_padded[i] == shape[i]:
                strides.append(strides_padded[i])  # Keep the original stride
            else:
                raise RuntimeError(f"Incorrect shape for broadcast: {shape}")
    
        # Create a new tensor with the broadcasted shape and updated strides
        result = TensorBase(self)
        result.ndim = len(shape)
        result.strides = tuple(strides)
        result.shape = tuple(shape)
        
        # Update the tensor metadata in the C backend
        _C.update_tensor(
            result.data,
            (ctypes.c_int * result.ndim)(*result.shape),
            (ctypes.c_int * result.ndim)(*result.strides),
            result.ndim
        )

        # Update the number of elements (numel) based on the broadcasted shape
        result.numel = result.data.contents.numel
        return result

    def broadcast_to_torch(self, shape):
        """Expands the tensor to the given shape using PyTorch-like broadcasting semantics."""
        strides = []
        result_ndim = len(shape)
        shape_padded = (1, ) * (result_ndim - self.ndim) + self.shape
        strides_padding = self.strides[0] * self.shape[0] if self.shape else 0
        strides_padded = (strides_padding, ) * (result_ndim - self.ndim) + self.strides
        
        zero_padding = False
        for i in range(result_ndim - 1, -1, -1):
            if shape_padded[i] != shape[i]:
                 if shape_padded[i] != 1:
                     raise RuntimeError(f"Incorrect shape for broadcast: {shape}")
                 else:
                    strides = [0] + strides
                    zero_padding = True if i < (result_ndim - self.ndim) else False
            else:
                if zero_padding:
                    strides = [0] + strides
                else:
                    strides = [strides_padded[i]] + strides
            
        result = TensorBase(self)
        result.ndim = len(shape)
        result.strides = tuple(strides)
        result.shape = tuple(shape)
        
        _C.update_tensor(
            result.data,
            (ctypes.c_int * result.ndim)(*result.shape),
            (ctypes.c_int * result.ndim)(*result.strides),
            result.ndim
        )

        result.numel = result.data.contents.numel
        return result
    
    @staticmethod
    def broadcast_shape(shape_left, shape_right):
        """
        Computes the broadcasted shape from two input shapes, following the broadcasting rules.

        Args:
            shape_left (tuple): Shape of the first tensor.
            shape_right (tuple): Shape of the second tensor.

        Returns:
            tuple: The resulting broadcasted shape.

        Raises:
            RuntimeError: If the two shapes are incompatible for broadcasting.
        """
        if shape_left == shape_right:
            return (shape_left)
        
        # Determine the maximum number of dimensions between the two shapes
        left_ndim = len(shape_left)
        right_ndim = len(shape_right)
        result_ndim = max(left_ndim, right_ndim)
        
        # Pad the shapes with 1s to match the maximum number of dimensions
        left_padded = (1, ) * (result_ndim - left_ndim) + shape_left
        right_padded = (1, ) * (result_ndim - right_ndim) + shape_right
        
        # Store the resulting shape
        result_shape = []

        # Iterate over padded shapes and compare corresponding axes
        for left_axis, right_axis in zip(left_padded, right_padded):
            if right_axis > left_axis:  # If the right axis is greater, broadcasting occurs for the left tensor
                if left_axis != 1:
                    raise RuntimeError(f"Shape {shape_left} must match {shape_right} at non-singleton dimensions")
                result_shape.append(right_axis)
            elif right_axis < left_axis:  # Broadcasting occurs for the right tensor
                if right_axis != 1:
                    raise RuntimeError(f"Shape {shape_left} must match {shape_right} at non-singleton dimensions")
                result_shape.append(left_axis)
            else:
                result_shape.append(left_axis)
        
        return tuple(result_shape)

    @staticmethod
    def empty(shape, dtype=None, device="cpu"):
        """
        Creates an uninitialized tensor with the specified shape, data type, and device.

        Args:
            shape (tuple | int): The shape of the tensor. 
            dtype (str, optional): Data type ('double' or 'long'). Defaults to 'double' if not provided.
            device (str, optional): The device on which to allocate the tensor ('cpu' or 'cuda'). Defaults to 'cpu'.

        Returns:
            TensorBase: An uninitialized tensor with the specified shape, data type, and device.

        Raises:
            RuntimeError: If any dimension in `shape` is negative.
        """
        if isinstance(shape, int):
            shape = (shape, )
        numel = 1
        for dim in shape:
            if dim >= 0:
                numel *= dim
            else:
                raise RuntimeError(f"Incorrect shape: {shape}")
        dtype = dtype if dtype is not None else "double"
        ndim = len(shape)
       
        data = _C.create_tensor(
            _C.create_storage(ctypes.c_int(ctypes.sizeof(dtypes[dtype][1]) * numel), # nbytes
                              ctypes.c_int(-1 if device =="cpu" else 0)              # device type
                             ),                    # storage
            (ctypes.c_int * ndim)(*shape),         # shape
            ctypes.c_int(ndim),                    # ndim
            ctypes.c_int(dtypes[dtype][0]),        # dtype code
        )
        return TensorBase(data)

    @staticmethod
    def zeros(shape, dtype=None, device="cpu"):
        """
        Creates a tensor filled with zeros.

        Args:
            shape (tuple | int): The shape of the tensor. 
            dtype (str, optional): Data type ('double' or 'long'). Defaults to 'double' if not provided.
            device (str, optional): The device on which to allocate the tensor ('cpu' or 'cuda'). Defaults to 'cpu'.

        Returns:
            TensorBase: A tensor of the specified shape filled with zeros.
        """
        result = TensorBase.empty(shape, dtype=dtype, device=device)
        result[:] = 0
        return result

    @staticmethod
    def ones(shape, dtype=None, device="cpu"):
        """
        Creates a tensor filled with ones.

        Args:
            shape (tuple | int): The shape of the tensor. 
            dtype (str, optional): Data type ('double' or 'long'). Defaults to 'double' if not provided.
            device (str, optional): The device on which to allocate the tensor ('cpu' or 'cuda'). Defaults to 'cpu'.

        Returns:
            TensorBase: A tensor of the specified shape filled with ones.
        """
        result = TensorBase.empty(shape, dtype=dtype, device=device)
        result[:] = 1
        return result

    @staticmethod
    def zeros_like(tensor):
        """Returns a tensor of the same shape as `tensor`, filled with zeros."""
        return TensorBase.zeros(shape=tensor.shape, dtype=tensor.dtype, device=tensor.device)

    @staticmethod
    def ones_like(tensor):
        """Returns a tensor of the same shape as `tensor`, filled with ones."""
        return TensorBase.ones(shape=tensor.shape, dtype=tensor.dtype, device=tensor.device)
    
    @staticmethod
    def arange(start=0, stop=None, step=1, device="cpu"):
        """
        Creates a one-dimensional tensor containing evenly spaced values in the specified range.

        Args:
            start (int, optional): The starting value of the range. Defaults to 0.
            stop (int): The end value of the range.
            step (int, optional): The spacing between values. Defaults to 1.
            device (str, optional): The device on which to allocate the tensor ('cpu' or 'cuda'). Defaults to 'cpu'.

        Returns:
            TensorBase: A one-dimensional tensor containing values in the specified range.

        Raises:
            AssertionError: If the range arguments are inconsistent.
        """
        if stop is None:
            stop = start
            start = 0
        assert step !=0 and -((stop - start) // -step) >= 0
        return TensorBase(_C.arange(ctypes.c_int(start), ctypes.c_int(stop), ctypes.c_int(step), ctypes.c_int(-1 if device =="cpu" else 0)))
    
    def expand_dims(self, axis):
        """
        Expands the dimensions of the tensor at the specified axis.

        Args:
            axis (tuple | int): The axis/axes along which to expand.
        
        Returns:
            TensorBase: A new tensor with expanded dimensions.
        """
        if isinstance(axis, int):
            axis = (axis, )  # Ensure axis is iterable
        
        # Convert negative indices to positive
        axis = [ax if ax >= 0 else ax + self.ndim + len(axis) for ax in axis]
        result_shape = list(self.shape)

        # Insert singleton dimensions
        for ax in sorted(axis):
            result_shape.insert(ax, 1)
        return self.reshape(result_shape)
    
    def squeeze(self, axis=None):
        """
        Removes singleton dimensions from the tensor.

        Args:
            axis (tuple | int | None): The axis/axes to squeeze. If None, all singleton dimensions are removed.

        Returns:
            TensorBase: A new tensor with squeezed dimensions.
        """
        if isinstance(axis, int):
            axis = (axis, ) # Ensure axis is iterable
        
        if axis is None:
            # If no axis is specified, remove all singleton dimensions
            axis = [i for i in range(len(self.shape)) if self.shape[i] == 1]
        else:
            # Convert negative indices to positive
            axis = [ax if ax >= 0 else ax + self.ndim for ax in axis]
        
        # Create a new shape excluding squeezed dimensions
        result_shape = [self.shape[i] for i in range(self.ndim) if i not in axis or self.shape[i] != 1]

        return self.reshape(result_shape)
    
    def swapaxes(self, axis1, axis2):
        """
        Swaps two axes of the tensor.

        Args:
            axis1 (int): First axis to swap.
            axis2 (int): Second axis to swap.

        Returns:
            TensorBase: A new tensor with the specified axes swapped.
        """
        # Swap shape and stride order
        result_shape = list(self.shape)
        result_strides = list(self.strides)
        result_shape[axis1], result_shape[axis2] = result_shape[axis2], result_shape[axis1]
        result_strides[axis1], result_strides[axis2] = result_strides[axis2], result_strides[axis1]
        
        # Create a new tensor with the updated shape and strides
        result = TensorBase(self)
        result.shape = tuple(result_shape)
        result.strides = tuple(result_strides)

        # Update the tensor metadata in the C backend
        _C.update_tensor(
            result.data,
            (ctypes.c_int * result.ndim)(*result.shape),
            (ctypes.c_int * result.ndim)(*result.strides),
            result.ndim
        )
        return result
    
    @property
    def T(self):
        """Returns the transposed tensor."""
        if self.ndim > 1:
            return self.swapaxes(-1, -2)
        else: 
            return self
    
    def sum(self, axis=None, keepdims=False):
        """
        Computes the sum of tensor elements along the specified axis.

        Args:
            axis (tuple | int | None): Axis or axes along which to sum. If None, sum over all axes.
            keepdims (bool): Whether to retain reduced dimensions as size 1.

        Returns:
            TensorBase: A new tensor containing the summed values.
        """
        if axis is None:
            axis = list(range(self.ndim))
        if isinstance(axis, int):
            axis = (axis, )

        axis = [ax if ax >= 0 else ax + self.ndim for ax in axis]

        res_data = _C.sum(
            self.data,
            (ctypes.c_int * len(axis))(*axis),
            ctypes.c_int(len(axis))
        )
        return TensorBase(res_data) if keepdims else TensorBase(res_data).squeeze(axis)

    def max(self, axis=None, keepdims=False):
        """
        Computes the maximum value of the tensor along the specified axis.

        Args:
            axis (tuple | int | None): Axis or axes along which to compute max. If None, find the global max.
            keepdims (bool): Whether to retain reduced dimensions as size 1.

        Returns:
            TensorBase: A new tensor containing the maximum values.
        """
        if axis is None:
            axis = list(range(self.ndim))
        if isinstance(axis, int):
            axis = (axis, )

        axis = [ax if ax >= 0 else ax + self.ndim for ax in axis]

        res_data = _C.max_t(
            self.data,
            (ctypes.c_int * len(axis))(*axis),
            ctypes.c_int(len(axis))
        )
        return TensorBase(res_data) if keepdims else TensorBase(res_data).squeeze(axis)


    def argmax(self, axis=None, keepdims=False):
        """
        Returns the indices of the maximum values along the specified axis.

        Args:
            axis (int | None): Axis along which to find the maximum index. If None, flattens the tensor before finding argmax.
            keepdims (bool): Whether to retain reduced dimensions as size 1.

        Returns:
            TensorBase: A new tensor with indices of maximum values.
        """
        if axis is None:
            axis = 0
            self = self.reshape((-1,))
        elif axis < 0:
            axis = axis + self.ndim

        res_data = _C.argmax(
            self.data,
            axis
        )
        return TensorBase(res_data) if keepdims else TensorBase(res_data).squeeze(axis)

    def __gt__(self, other):
        """
        Element-wise greater than comparison.

        Args:
            other (TensorBase | int | float): The value to compare with.

        Returns:
            TensorBase: A tensor with integer values (1 for True, 0 for False).
        """
        if isinstance(other, float):
            other = TensorBase(other, device=self.device)
        elif isinstance(other, int):
            other = TensorBase(other, dtype="long", device=self.device)
        # Broadcast if shapes are different
        if self.shape != other.shape:
            broadcast_shape = TensorBase.broadcast_shape(self.shape, other.shape)
            if self.shape != broadcast_shape:
                self = self.broadcast_to(broadcast_shape)
            if other.shape != broadcast_shape:
                other = other.broadcast_to(broadcast_shape)

        res_data = _C.gt(self.data, other.data)
        return TensorBase(res_data)

    def __eq__(self, other):
        """
        Element-wise equality comparison.

        Args:
            other (TensorBase | int | float): The value to compare with.

        Returns:
            TensorBase: A tensor with integer values (1 for True, 0 for False).
        """
        if isinstance(other, float):
            other = TensorBase(other, device=self.device)
        elif isinstance(other, int):
            other = TensorBase(other, dtype="long", device=self.device)
         # Broadcast if shapes are different
        if self.shape != other.shape:
            broadcast_shape = TensorBase.broadcast_shape(self.shape, other.shape)
            if self.shape != broadcast_shape:
                self = self.broadcast_to(broadcast_shape)
            if other.shape != broadcast_shape:
                other = other.broadcast_to(broadcast_shape)

        res_data = _C.eq(self.data, other.data)
        return TensorBase(res_data)

    def __add__(self, other):
        """
        Element-wise addition.

        Args:
            other (TensorBase | int | float): The value to add.

        Returns:
            TensorBase: Result of element-wise addition.
        """
        if isinstance(other, float):
            other = TensorBase(other, device=self.device)
        elif isinstance(other, int):
            other = TensorBase(other, dtype="long", device=self.device)
         # Broadcast if shapes are different
        if self.shape != other.shape:
            broadcast_shape = TensorBase.broadcast_shape(self.shape, other.shape)
            if self.shape != broadcast_shape:
                self = self.broadcast_to(broadcast_shape)
            if other.shape != broadcast_shape:
                other = other.broadcast_to(broadcast_shape)

        res_data = _C.add(self.data, other.data)
        return TensorBase(res_data)

    def __mul__(self, other):
        """
        Element-wise multiplication.

        Args:
            other (TensorBase | int | float): The value to multiply.

        Returns:
            TensorBase: Result of element-wise multiplication.
        """
        if isinstance(other, float):
            other = TensorBase(other, device=self.device)
        elif isinstance(other, int):
            other = TensorBase(other, dtype="long", device=self.device)
         # Broadcast if shapes are different
        if self.shape != other.shape:
            broadcast_shape = TensorBase.broadcast_shape(self.shape, other.shape)
            if self.shape != broadcast_shape:
                self = self.broadcast_to(broadcast_shape)
            if other.shape != broadcast_shape:
                other = other.broadcast_to(broadcast_shape)

        res_data = _C.mul(self.data, other.data)
        return TensorBase(res_data)

    def maximum(self, other):
        """
        Element-wise maximum.

        Args:
            other (TensorBase | int | float): The value to compare.

        Returns:
            TensorBase: Element-wise maximum.
        """
        if isinstance(other, float):
            other = TensorBase(other, device=self.device)
        elif isinstance(other, int):
            other = TensorBase(other, dtype="long", device=self.device)
         # Broadcast if shapes are different
        if self.shape != other.shape:
            broadcast_shape = TensorBase.broadcast_shape(self.shape, other.shape)
            if self.shape != broadcast_shape:
                self = self.broadcast_to(broadcast_shape)
            if other.shape != broadcast_shape:
                other = other.broadcast_to(broadcast_shape)

        res_data = _C.maximum(self.data, other.data)
        return TensorBase(res_data)

    def __matmul__(self, other):
        """
        Matrix multiplication.

        Args:
            other (TensorBase): The right-hand side tensor in the matrix multiplication.

        Returns:
            TensorBase: Result of the matrix multiplication.
        """
         # Determine if extra dimensions are needed
        expand_self = other.ndim > 1
        expand_other = self.ndim > 1
        self = self.expand_dims(-2) if expand_self else self
        other = other.T.expand_dims(-3 if other.ndim > 1 else -2) if expand_other else other.T

        # Broadcast if shapes are different
        if self.shape != other.shape:
            broadcast_shape = TensorBase.broadcast_shape(self.shape, other.shape)
            if self.shape != broadcast_shape:
                self = self.broadcast_to(broadcast_shape)
            if other.shape != broadcast_shape:
                other = other.broadcast_to(broadcast_shape)
        
        # Perform element-wise multiplication and reduction along the last dimension
        res_data = _C.mul_reduce(self.data, other.data, self.ndim - 1)
        # Remove the last singleton dimension to match the expected output shape
        result = TensorBase(res_data).squeeze(-1)
        return result
    
    def __pow__(self, other):
        """
        Raises each element of the tensor to the power of other.
        
        Args:
            other (int | float): The exponent.
        
        Returns:
            TensorBase: A new tensor with elements raised to the power of other.
        """
        res_data = _C.pow_t(self.data, float(other))
        return TensorBase(res_data)
    
    def add_at(self, idx, other):
        """
        Accumulates values into the tensor at the specified indices in place.

        Args:
            idx (TensorBase): A tensor containing the indices where values will be added.
            other (TensorBase | int | float): The value to add.

        Raises:
            RuntimeError: If `other` cannot be broadcast to the shape of `idx`.
        """
        if isinstance(other, float):
            other = TensorBase(other, device=self.device)
        elif isinstance(other, int):
            other = TensorBase(other, dtype="long", device=self.device)
         # Broadcast if shapes are different
        if idx.shape != other.shape:
            broadcast_shape = TensorBase.broadcast_shape(idx.shape, other.shape)
            if idx.shape != broadcast_shape:
                raise RuntimeError(f"Incorrect shape: {other.shape}")
            other = other.broadcast_to(broadcast_shape)
        if idx.device != self.device:
            idx = idx.to(self.device)
        if other.device != self.device:
            other = other.to(self.device)

        _C.add_at(self.data, idx.data, other.data)
    
    def uniform_(self, a, b):
        """Fills the tensor with values sampled uniformly from [a, b) in place."""
        _C.uniform(self.data, a, b)
        return self

    def exp(self):
        """Computes the element-wise exponential of the tensor."""
        res_data = _C.exp_t(self.data)
        return TensorBase(res_data)

    def log(self):
        """Computes the element-wise natural logarithm of the tensor."""
        res_data = _C.log_t(self.data)
        return TensorBase(res_data)

    def tanh(self):
        """Computes the element-wise hyperbolic tangent of the tensor."""
        res_data = _C.tanh_t(self.data)
        return TensorBase(res_data)
    
    def contiguous(self):
        """Returns a contiguous tensor in memory. If the tensor is contiguous, returns itself."""
        return self if self.is_contiguous() else TensorBase(_C.contiguous(self.data))
    
    def is_contiguous(self):
        """Checks if the tensor is stored in a contiguous memory layout."""
        return _C.is_contiguous(self.data)
    
    def item(self):
        """
        Converts a single-element tensor into a scalar.
        
        Returns:
            (int | float): The scalar value of the tensor.
        """
        return dtypes[self.dtype][1].from_address(_C.get_item(self.data, 0)).value

    def __del__(self):
        """Handles safe deletion of the tensor."""
        _C.delete_tensor(self.data)

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