import ugrad
import torch
import numpy as np


def test_conv2d():
    x = np.arange(2 * 3 * 10 * 20).reshape((2, 3, 10, 20)) * 0.1
    k = np.arange(5 * 3 * 2 * 3).reshape(5, 3, 2, 3) * 0.1
    b = np.arange(5) * 0.1
    xx = ugrad.Tensor(x)
    kk = ugrad.Tensor(k, requires_grad=True)
    bb = ugrad.Tensor(b, requires_grad=True)

    f = ugrad.nn.functional.conv2d(xx, kk, bias=bb, stride=1, padding=0, dilation=1)
    
    grad_inp = np.arange(f.data.size).reshape(f.data.shape) * 0.1
    
    f.backward(grad_inp)
    fut, kut, but = f, kk, bb

    xx = torch.tensor(x, dtype=torch.float64)
    kk = torch.tensor(k, requires_grad=True, dtype=torch.float64)
    bb = torch.tensor(b, requires_grad=True, dtype=torch.float64)

    f = torch.nn.functional.conv2d(xx, kk, bias=bb, stride=1, padding=0, dilation=1)
    
    f.backward(torch.tensor(grad_inp))
    fpt, kpt, bpt = f, kk, bb

    tol = 1e-6
    # forward
    assert (np.abs(fut.data - fpt.data.numpy()) < tol).all()
    # backward
    assert (np.abs(kut.grad - kpt.grad.numpy()) < tol).all()
    assert (np.abs(but.grad - bpt.grad.numpy()) < tol).all()


def test_conv2d_pad():
    x = np.arange(2 * 3 * 10 * 20).reshape((2, 3, 10, 20)) * 0.1
    k = np.arange(5 * 3 * 2 * 3).reshape(5, 3, 2, 3) * 0.1
    b = np.arange(5) * 0.1
    xx = ugrad.Tensor(x)
    kk = ugrad.Tensor(k, requires_grad=True)
    bb = ugrad.Tensor(b, requires_grad=True)

    f = ugrad.nn.functional.conv2d(xx, kk, bias=bb, stride=1, padding=1, dilation=1)
    
    grad_inp = np.arange(f.data.size).reshape(f.data.shape) * 0.1
    
    f.backward(grad_inp)
    fut, kut, but = f, kk, bb

    xx = torch.tensor(x, dtype=torch.float64)
    kk = torch.tensor(k, requires_grad=True, dtype=torch.float64)
    bb = torch.tensor(b, requires_grad=True, dtype=torch.float64)

    f = torch.nn.functional.conv2d(xx, kk, bias=bb, stride=1, padding=1, dilation=1)
    
    f.backward(torch.tensor(grad_inp))
    fpt, kpt, bpt = f, kk, bb

    tol = 1e-6
    # forward
    assert (np.abs(fut.data - fpt.data.numpy()) < tol).all()
    # backward
    assert (np.abs(kut.grad - kpt.grad.numpy()) < tol).all()
    assert (np.abs(but.grad - bpt.grad.numpy()) < tol).all()


def test_conv2d_pad_dilation():
    x = np.arange(2 * 3 * 10 * 20).reshape((2, 3, 10, 20)) * 0.1
    k = np.arange(5 * 3 * 2 * 3).reshape(5, 3, 2, 3) * 0.1
    b = np.arange(5) * 0.1
    xx = ugrad.Tensor(x)
    kk = ugrad.Tensor(k, requires_grad=True)
    bb = ugrad.Tensor(b, requires_grad=True)

    f = ugrad.nn.functional.conv2d(xx, kk, bias=bb, stride=1, padding=2, dilation=2)
    
    grad_inp = np.arange(f.data.size).reshape(f.data.shape) * 0.1
    
    f.backward(grad_inp)
    fut, kut, but = f, kk, bb

    xx = torch.tensor(x, dtype=torch.float64)
    kk = torch.tensor(k, requires_grad=True, dtype=torch.float64)
    bb = torch.tensor(b, requires_grad=True, dtype=torch.float64)

    f = torch.nn.functional.conv2d(xx, kk, bias=bb, stride=1, padding=2, dilation=2)
    
    f.backward(torch.tensor(grad_inp))
    fpt, kpt, bpt = f, kk, bb

    tol = 1e-6
    # forward
    assert (np.abs(fut.data - fpt.data.numpy()) < tol).all()
    # backward
    assert (np.abs(kut.grad - kpt.grad.numpy()) < tol).all()
    assert (np.abs(but.grad - bpt.grad.numpy()) < tol).all()


def test_avg_pool2d():
    x = np.arange(2 * 3 * 10 * 20).reshape((2, 3, 10, 20)) * 0.1
    xx = ugrad.Tensor(x, requires_grad=True)

    f = ugrad.nn.functional.avg_pool2d(xx, kernel_size=2, stride=None, padding=1)

    grad_inp = np.arange(f.data.size).reshape(f.data.shape) * 0.1

    f.backward(grad_inp)
    fut, xxut = f, xx

    xx = torch.tensor(x, requires_grad=True, dtype=torch.float64)

    f = torch.nn.functional.avg_pool2d(xx, kernel_size=2, stride=None, padding=1)
    
    f.backward(torch.tensor(grad_inp))
    fpt, xxpt = f, xx

    tol = 1e-6
    # forward
    assert (np.abs(fut.data - fpt.data.numpy()) < tol).all()
    # backward
    assert (np.abs(xxut.grad - xxpt.grad.numpy()) < tol).all()

    