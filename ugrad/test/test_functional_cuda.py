import ugrad
import torch
import numpy as np


def test_conv2d_cuda():
    x = np.arange(2 * 3 * 10 * 20).reshape((2, 3, 10, 20)) * 0.1
    k = np.arange(5 * 3 * 2 * 3).reshape(5, 3, 2, 3) * 0.1
    b = np.arange(5) * 0.1
    xx = ugrad.Tensor(x, requires_grad=True, device="cuda")
    kk = ugrad.Tensor(k, requires_grad=True, device="cuda")
    bb = ugrad.Tensor(b, requires_grad=True, device="cuda")

    f = ugrad.nn.functional.conv2d(xx, kk, bias=bb, stride=1, padding=0, dilation=1)
    
    fdnp = f.data.to("cpu").numpy()

    grad_inp = np.arange(fdnp.size).reshape(fdnp.shape) * 0.1
    
    f.backward(ugrad.TensorBase(grad_inp, device="cuda"))
    fut, kut, xut, but = f, kk, xx, bb

    xx = torch.tensor(x, requires_grad=True, dtype=torch.float64, device="cuda")
    kk = torch.tensor(k, requires_grad=True, dtype=torch.float64, device="cuda")
    bb = torch.tensor(b, requires_grad=True, dtype=torch.float64, device="cuda")

    f = torch.nn.functional.conv2d(xx, kk, bias=bb, stride=1, padding=0, dilation=1)
    
    f.backward(torch.tensor(grad_inp, device="cuda"))
    fpt, kpt, xpt, bpt = f, kk, xx, bb

    tol = 1e-6
    # forward
    assert (np.abs(fut.data.to("cpu").numpy() - fpt.data.to("cpu").numpy()) < tol).all()
    # backward
    assert (np.abs(kut.grad.to("cpu").numpy() - kpt.grad.to("cpu").numpy()) < tol).all()
    assert (np.abs(xut.grad.to("cpu").numpy() - xpt.grad.to("cpu").numpy()) < tol).all()
    assert (np.abs(but.grad.to("cpu").numpy() - bpt.grad.to("cpu").numpy()) < tol).all()


def test_conv2d_stride_cuda():
    x = np.arange(2 * 3 * 10 * 20).reshape((2, 3, 10, 20)) * 0.1
    k = np.arange(5 * 3 * 2 * 3).reshape(5, 3, 2, 3) * 0.1
    b = np.arange(5) * 0.1
    xx = ugrad.Tensor(x, requires_grad=True, device="cuda")
    kk = ugrad.Tensor(k, requires_grad=True, device="cuda")
    bb = ugrad.Tensor(b, requires_grad=True, device="cuda")

    f = ugrad.nn.functional.conv2d(xx, kk, bias=bb, stride=2, padding=0, dilation=1)
    
    fdnp = f.data.to("cpu").numpy()

    grad_inp = np.arange(fdnp.size).reshape(fdnp.shape) * 0.1
    
    f.backward(ugrad.TensorBase(grad_inp, device="cuda"))
    fut, kut, xut, but = f, kk, xx, bb

    xx = torch.tensor(x, requires_grad=True, dtype=torch.float64, device="cuda")
    kk = torch.tensor(k, requires_grad=True, dtype=torch.float64, device="cuda")
    bb = torch.tensor(b, requires_grad=True, dtype=torch.float64, device="cuda")

    f = torch.nn.functional.conv2d(xx, kk, bias=bb, stride=2, padding=0, dilation=1)
    
    f.backward(torch.tensor(grad_inp, device="cuda"))
    fpt, kpt, xpt, bpt = f, kk, xx, bb

    tol = 1e-6
    # forward
    assert (np.abs(fut.data.to("cpu").numpy() - fpt.data.to("cpu").numpy()) < tol).all()
    # backward
    assert (np.abs(kut.grad.to("cpu").numpy() - kpt.grad.to("cpu").numpy()) < tol).all()
    assert (np.abs(xut.grad.to("cpu").numpy() - xpt.grad.to("cpu").numpy()) < tol).all()
    assert (np.abs(but.grad.to("cpu").numpy() - bpt.grad.to("cpu").numpy()) < tol).all()


def test_conv2d_pad_cuda():
    x = np.arange(2 * 3 * 10 * 20).reshape((2, 3, 10, 20)) * 0.1
    k = np.arange(5 * 3 * 2 * 3).reshape(5, 3, 2, 3) * 0.1
    b = np.arange(5) * 0.1
    xx = ugrad.Tensor(x, requires_grad=True, device="cuda")
    kk = ugrad.Tensor(k, requires_grad=True, device="cuda")
    bb = ugrad.Tensor(b, requires_grad=True, device="cuda")

    f = ugrad.nn.functional.conv2d(xx, kk, bias=bb, stride=1, padding=1, dilation=1)
    
    fdnp = f.data.to("cpu").numpy()

    grad_inp = np.arange(fdnp.size).reshape(fdnp.shape) * 0.1
    
    f.backward(ugrad.TensorBase(grad_inp, device="cuda"))
    fut, kut, xut, but = f, kk, xx, bb

    xx = torch.tensor(x, requires_grad=True, dtype=torch.float64, device="cuda")
    kk = torch.tensor(k, requires_grad=True, dtype=torch.float64, device="cuda")
    bb = torch.tensor(b, requires_grad=True, dtype=torch.float64, device="cuda")

    f = torch.nn.functional.conv2d(xx, kk, bias=bb, stride=1, padding=1, dilation=1)
    
    f.backward(torch.tensor(grad_inp, device="cuda"))
    fpt, kpt, xpt, bpt = f, kk, xx, bb

    tol = 1e-6
    # forward
    assert (np.abs(fut.data.to("cpu").numpy() - fpt.data.to("cpu").numpy()) < tol).all()
    # backward
    assert (np.abs(kut.grad.to("cpu").numpy() - kpt.grad.to("cpu").numpy()) < tol).all()
    assert (np.abs(xut.grad.to("cpu").numpy() - xpt.grad.to("cpu").numpy()) < tol).all()
    assert (np.abs(but.grad.to("cpu").numpy() - bpt.grad.to("cpu").numpy()) < tol).all()


def test_conv2d_pad_dilation_cuda():
    x = np.arange(2 * 3 * 10 * 20).reshape((2, 3, 10, 20)) * 0.1
    k = np.arange(5 * 3 * 2 * 3).reshape(5, 3, 2, 3) * 0.1
    b = np.arange(5) * 0.1
    xx = ugrad.Tensor(x, requires_grad=True, device="cuda")
    kk = ugrad.Tensor(k, requires_grad=True, device="cuda")
    bb = ugrad.Tensor(b, requires_grad=True, device="cuda")

    f = ugrad.nn.functional.conv2d(xx, kk, bias=bb, stride=1, padding=2, dilation=2)
    
    fdnp = f.data.to("cpu").numpy()

    grad_inp = np.arange(fdnp.size).reshape(fdnp.shape) * 0.1
    
    f.backward(ugrad.TensorBase(grad_inp, device="cuda"))
    fut, kut, xut, but = f, kk, xx, bb

    xx = torch.tensor(x, requires_grad=True, dtype=torch.float64, device="cuda")
    kk = torch.tensor(k, requires_grad=True, dtype=torch.float64, device="cuda")
    bb = torch.tensor(b, requires_grad=True, dtype=torch.float64, device="cuda")

    f = torch.nn.functional.conv2d(xx, kk, bias=bb, stride=1, padding=2, dilation=2)
    
    f.backward(torch.tensor(grad_inp, device="cuda"))
    fpt, kpt, xpt, bpt = f, kk, xx, bb

    tol = 1e-6
    # forward
    assert (np.abs(fut.data.to("cpu").numpy() - fpt.data.to("cpu").numpy()) < tol).all()
    # backward
    assert (np.abs(kut.grad.to("cpu").numpy() - kpt.grad.to("cpu").numpy()) < tol).all()
    assert (np.abs(xut.grad.to("cpu").numpy() - xpt.grad.to("cpu").numpy()) < tol).all()
    assert (np.abs(but.grad.to("cpu").numpy() - bpt.grad.to("cpu").numpy()) < tol).all()


def test_avg_pool2d_cuda():
    x = np.arange(2 * 3 * 10 * 20).reshape((2, 3, 10, 20)) * 0.1
    xx = ugrad.Tensor(x, requires_grad=True, device="cuda")

    f = ugrad.nn.functional.avg_pool2d(xx, kernel_size=2, stride=None, padding=1)

    fdnp = f.data.to("cpu").numpy()

    grad_inp = np.arange(fdnp.size).reshape(fdnp.shape) * 0.1

    f.backward(ugrad.TensorBase(grad_inp, device="cuda"))
    fut, xut = f, xx

    xx = torch.tensor(x, requires_grad=True, dtype=torch.float64, device="cuda")

    f = torch.nn.functional.avg_pool2d(xx, kernel_size=2, stride=None, padding=1)
    
    f.backward(torch.tensor(grad_inp, device="cuda"))
    fpt, xpt = f, xx

    tol = 1e-6
    # forward
    assert (np.abs(fut.data.to("cpu").numpy() - fpt.data.to("cpu").numpy()) < tol).all()
    # backward
    assert (np.abs(xut.grad.to("cpu").numpy() - xpt.grad.to("cpu").numpy()) < tol).all()


def test_avg_pool2d_stride_cuda():
    x = np.arange(2 * 3 * 10 * 20).reshape((2, 3, 10, 20)) * 0.1
    xx = ugrad.Tensor(x, requires_grad=True, device="cuda")

    f = ugrad.nn.functional.avg_pool2d(xx, kernel_size=2, stride=1, padding=1)

    fdnp = f.data.to("cpu").numpy()

    grad_inp = np.arange(fdnp.size).reshape(fdnp.shape) * 0.1

    f.backward(ugrad.TensorBase(grad_inp, device="cuda"))
    fut, xut = f, xx

    xx = torch.tensor(x, requires_grad=True, dtype=torch.float64, device="cuda")

    f = torch.nn.functional.avg_pool2d(xx, kernel_size=2, stride=1, padding=1)
    
    f.backward(torch.tensor(grad_inp, device="cuda"))
    fpt, xpt = f, xx

    tol = 1e-6
    # forward
    assert (np.abs(fut.data.to("cpu").numpy() - fpt.data.to("cpu").numpy()) < tol).all()
    # backward
    assert (np.abs(xut.grad.to("cpu").numpy() - xpt.grad.to("cpu").numpy()) < tol).all()


def test_max_pool2d_cuda():
    x = np.arange(2 * 3 * 10 * 20).reshape((2, 3, 10, 20)) * 0.1
    xx = ugrad.Tensor(x, requires_grad=True, device="cuda")

    f = ugrad.nn.functional.max_pool2d(xx, kernel_size=2, stride=None, padding=0)

    fdnp = f.data.to("cpu").numpy()

    grad_inp = np.arange(fdnp.size).reshape(fdnp.shape) * 0.1

    f.backward(ugrad.TensorBase(grad_inp, device="cuda"))
    fut, xut = f, xx

    xx = torch.tensor(x, requires_grad=True, dtype=torch.float64, device="cuda")

    f = torch.nn.functional.max_pool2d(xx, kernel_size=2, stride=None, padding=0)
    
    f.backward(torch.tensor(grad_inp, device="cuda"))
    fpt, xpt = f, xx

    tol = 1e-6
    # forward
    assert (np.abs(fut.data.to("cpu").numpy() - fpt.data.to("cpu").numpy()) < tol).all()
    # backward
    assert (np.abs(xut.grad.to("cpu").numpy() - xpt.grad.to("cpu").numpy()) < tol).all()
