import ugrad
import torch
import numpy as np


def test_sum1D():

    x = ugrad.Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x.sum()

    y.backward()
    xut, yut = x, y

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True, dtype=torch.float64)
    y = x.sum()
    
    y.backward()
    xpt, ypt = x, y

    # forward
    assert (yut.data == ypt.data.numpy()).all()
    # backward
    assert (xut.grad == xpt.grad.numpy()).all()


def test_sum2D_row():

    x = ugrad.Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    y = x.sum()

    y.backward()
    xut, yut = x, y

    x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True, dtype=torch.float64)
    y = x.sum()

    y.backward()
    xpt, ypt = x, y

    # forward
    assert (yut.data == ypt.data.numpy()).all()
    # backward
    assert (xut.grad == xpt.grad.numpy()).all()


def test_sum2D_col():

    x = ugrad.Tensor([[1.0], [2.0], [3.0]], requires_grad=True)
    y = x.sum()

    y.backward()
    xut, yut = x, y

    x = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True, dtype=torch.float64)
    y = x.sum()

    y.backward()
    xpt, ypt = x, y

    # forward
    assert (yut.data == ypt.data.numpy()).all()
    # backward
    assert (xut.grad == xpt.grad.numpy()).all()


def test_sum2D_axis0():

    xx = np.arange(2*3).reshape(2, 3)
    yy = np.arange(3)

    x = ugrad.Tensor(xx, requires_grad=True)
    y = ugrad.Tensor(yy, requires_grad=True)
    z = x.sum(dim=0) @ y

    z.backward()
    xut, yut, zut = x, y, z

    x = torch.tensor(xx, requires_grad=True, dtype=torch.float64)
    y = torch.tensor(yy, requires_grad=True, dtype=torch.float64)
    z = x.sum(dim=0) @ y

    z.backward()
    xpt, ypt, zpt = x, y, z

    # forward
    assert (zut.data == zpt.data.numpy()).all()
    # backward
    assert (xut.grad == xpt.grad.numpy()).all()
    assert (yut.grad == ypt.grad.numpy()).all()


def test_sum2D_axis1():

    xx = np.arange(2*3).reshape(2, 3)
    yy = np.arange(2)

    x = ugrad.Tensor(xx, requires_grad=True)
    y = ugrad.Tensor(yy, requires_grad=True)
    z = x.sum(dim=1) @ y

    z.backward()
    xut, yut, zut = x, y, z

    x = torch.tensor(xx, requires_grad=True, dtype=torch.float64)
    y = torch.tensor(yy, requires_grad=True, dtype=torch.float64)
    z = x.sum(dim=1) @ y

    z.backward()
    xpt, ypt, zpt = x, y, z

    # forward
    assert (zut.data == zpt.data.numpy()).all()
    # backward
    assert (xut.grad == xpt.grad.numpy()).all()
    assert (yut.grad == ypt.grad.numpy()).all()


def test_matmul():

    x = ugrad.Tensor([[1.0], 
                      [2.0], 
                      [3.0]], requires_grad=True)
    y = ugrad.Tensor([[3.0], 
                      [5.0], 
                      [7.0]], requires_grad=True)
    z = x.T @ y

    z.backward()
    xut, yut, zut = x, y, z

    x = torch.tensor([[1.0], 
                      [2.0], 
                      [3.0]], requires_grad=True, dtype=torch.float64)
    y = torch.tensor([[3.0], 
                      [5.0], 
                      [7.0]], requires_grad=True, dtype=torch.float64)
    z = x.T @ y
    
    z.backward()
    xpt, ypt, zpt = x, y, z

    # forward
    assert (zut.data == zpt.data.numpy()).all()
    # backward
    assert (xut.grad == xpt.grad.numpy()).all()
    assert (yut.grad == ypt.grad.numpy()).all()


def test_matmul2():

    xx = np.arange(5*2*3).reshape(5, 2, 3) 
    yy = np.arange(5*3*2).reshape(5, 3, 2)

    x = ugrad.Tensor(xx, requires_grad=True)
    y = ugrad.Tensor(yy, requires_grad=True)
    z = (x @ y).sum()

    z.backward()
    xut, yut, zut = x, y, z

    x = torch.tensor(xx, requires_grad=True, dtype=torch.float64)
    y = torch.tensor(yy, requires_grad=True, dtype=torch.float64)
    z = (x @ y).sum()
    
    z.backward()
    xpt, ypt, zpt = x, y, z

    # forward
    assert (zut.data == zpt.data.numpy()).all()
    # backward
    assert (xut.grad == xpt.grad.numpy()).all()
    assert (yut.grad == ypt.grad.numpy()).all()


def test_matmul_broadcast():

    x = ugrad.Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = ugrad.Tensor([3.0, 5.0, 7.0], requires_grad=True)
    z = x @ y

    z.backward()
    xut, yut, zut = x, y, z

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True, dtype=torch.float64)
    y = torch.tensor([3.0, 5.0, 7.0], requires_grad=True, dtype=torch.float64)
    z = x @ y
    
    z.backward()
    xpt, ypt, zpt = x, y, z

    # forward
    assert (zut.data == zpt.data.numpy()).all()
    # backward
    assert (xut.grad == xpt.grad.numpy()).all()
    assert (yut.grad == ypt.grad.numpy()).all()


def test_matmul_broadcast2():

    x = ugrad.Tensor([1.0, 2.0], requires_grad=True)
    y = ugrad.Tensor([[3.0, 5.0, 7.0],
                      [-3.0, -1.0, 9.0]], requires_grad=True)
    c = ugrad.Tensor([13.0, -2.0, 5.0], requires_grad=True)
    z = (x @ y) @ c

    z.backward()
    xut, yut, cut, zut = x, y, c, z

    x = torch.tensor([1.0, 2.0], requires_grad=True, dtype=torch.float64)
    y = torch.tensor([[3.0, 5.0, 7.0],
                      [-3.0, -1.0, 9.0]], requires_grad=True, dtype=torch.float64)
    c = torch.tensor([13.0, -2.0, 5.0], requires_grad=True, dtype=torch.float64)
    z = (x @ y) @ c
    
    z.backward()
    xpt, ypt, cpt, zpt = x, y, c, z

    # forward
    assert (zut.data == zpt.data.numpy()).all()
    # backward
    assert (xut.grad == xpt.grad.numpy()).all()
    assert (yut.grad == ypt.grad.numpy()).all()
    assert (cut.grad == cpt.grad.numpy()).all()


def test_matmul_broadcast3():

    x = ugrad.Tensor([1.0, 2.0], requires_grad=True)
    y = ugrad.Tensor([[3.0, 5.0, 7.0],
                      [-3.0, -1.0, 9.0]], requires_grad=True)
    c = ugrad.Tensor([13.0, -2.0, 5.0], requires_grad=True)
    z = (y @ c) @ x

    z.backward()
    xut, yut, cut, zut = x, y, c, z

    x = torch.tensor([1.0, 2.0], requires_grad=True, dtype=torch.float64)
    y = torch.tensor([[3.0, 5.0, 7.0],
                      [-3.0, -1.0, 9.0]], requires_grad=True, dtype=torch.float64)
    c = torch.tensor([13.0, -2.0, 5.0], requires_grad=True, dtype=torch.float64)
    z = (y @ c) @ x
    
    z.backward()
    xpt, ypt, cpt, zpt = x, y, c, z

    # forward
    assert (zut.data == zpt.data.numpy()).all()
    # backward
    assert (xut.grad == xpt.grad.numpy()).all()
    assert (yut.grad == ypt.grad.numpy()).all()
    assert (cut.grad == cpt.grad.numpy()).all()


def test_matmul_broadcast4():

    aa = np.arange(7*1*5*2*3).reshape(7, 1, 5, 2, 3) * 0.1
    xx = np.arange(7*1*1*2*3).reshape(7, 1, 1, 2, 3) * 1.
    yy = np.arange(6*1*3*2).reshape(6, 1, 3, 2) * 1.
    cc = np.array([1, 2]) * 1.
    dd = np.array([3, 4]) * 0.1
    ee = np.arange(7 * 5 * 3).reshape(7, 5, 3) * 1.
    ff = np.arange(3) * 1.
    gg = np.ones((6, )) * 0.01
    hh = np.arange(7) * 1.

    a = ugrad.Tensor(aa, requires_grad=True)
    x = ugrad.Tensor(xx, requires_grad=True)
    y = ugrad.Tensor(yy, requires_grad=True)
    c = ugrad.Tensor(cc, requires_grad=True)
    d = ugrad.Tensor(dd, requires_grad=True)
    e = ugrad.Tensor(ee, requires_grad=True)
    f = ugrad.Tensor(ff, requires_grad=True)
    g = ugrad.Tensor(gg, requires_grad=True)
    h = ugrad.Tensor(hh, requires_grad=True)

    z = (a * x) @ y @ c @ d @ e @ f @ g @ h

    z.backward()
    xut, yut, cut, dut, eut, fut, gut, hut, zut = x, y, c, d, e, f, g, h, z

    a = torch.tensor(aa, requires_grad=True, dtype=torch.float64)
    x = torch.tensor(xx, requires_grad=True, dtype=torch.float64)
    y = torch.tensor(yy, requires_grad=True, dtype=torch.float64)
    c = torch.tensor(cc, requires_grad=True, dtype=torch.float64)
    d = torch.tensor(dd, requires_grad=True, dtype=torch.float64)
    e = torch.tensor(ee, requires_grad=True, dtype=torch.float64)
    f = torch.tensor(ff, requires_grad=True, dtype=torch.float64)
    g = torch.tensor(gg, requires_grad=True, dtype=torch.float64)
    h = torch.tensor(hh, requires_grad=True, dtype=torch.float64)

    z = (a * x) @ y @ c @ d @ e @ f @ g @ h
    
    z.backward()
    xpt, ypt, cpt, dpt, ept, fpt, gpt, hpt, zpt = x, y, c, d, e, f, g, h, z

    tol = 1e-6
    # forward
    assert (np.abs(zut.data - zpt.data.numpy()) < tol).all()

    # backward
    assert (np.abs(xut.grad - xpt.grad.numpy()) < tol).all()
    assert (np.abs(yut.grad - ypt.grad.numpy()) < tol).all()
    assert (np.abs(cut.grad - cpt.grad.numpy()) < tol).all()
    assert (np.abs(dut.grad - dpt.grad.numpy()) < tol).all()
    assert (np.abs(eut.grad - ept.grad.numpy()) < tol).all()
    assert (np.abs(fut.grad - fpt.grad.numpy()) < tol).all()
    assert (np.abs(gut.grad - gpt.grad.numpy()) < tol).all()
    assert (np.abs(hut.grad - hpt.grad.numpy()) < tol).all()


def test_liear_mse():

    x = ugrad.Tensor([[1.0, 2.0], 
                      [4.0, 5.0], 
                      [6.0, 7.0]], requires_grad=True)
    W = ugrad.Tensor([[0.5, -1.0]], requires_grad=True)
    y = ugrad.Tensor([[1.0], 
                      [-1.0], 
                      [0.0]], requires_grad=True)
    z = ((x @ W.T - y)**2).sum() / 3

    z.backward()
    xut, Wut, yut, zut = x, W, y, z

    x = torch.tensor([[1.0, 2.0], 
                      [4.0, 5.0], 
                      [6.0, 7.0]], requires_grad=True, dtype=torch.float64)
    W = torch.tensor([[0.5, -1.0]], requires_grad=True, dtype=torch.float64)
    y = torch.tensor([[1.0], 
                      [-1.0], 
                      [0.0]], requires_grad=True, dtype=torch.float64)
    z = ((x @ W.T - y)**2).sum() / 3
    
    z.backward()
    xpt, Wpt, ypt, zpt = x, W, y, z

    # forward
    assert (zut.data == zpt.data.numpy()).all()
    # backward
    assert (xut.grad == xpt.grad.numpy()).all()
    assert (Wut.grad == Wpt.grad.numpy()).all()
    assert (yut.grad == ypt.grad.numpy()).all()


def test_liear_bce():

    x = ugrad.Tensor([[1.0, 2.0], 
                      [4.0, 5.0], 
                      [6.0, 7.0]], requires_grad=True)
    W = ugrad.Tensor([[0.5, -1.0]], requires_grad=True)
    y = ugrad.Tensor([[1.0], 
                      [1.0], 
                      [0.0]], requires_grad=True)
    # sum([-(y[0] * y_hat[0].log() + (1 - y[0]) * (1 - y_hat[0]).log()) for y_hat, y in zip(preds, targets)]) / len(targets)
    z = (x @ W.T).sigmoid()
    a = -(y * z.log() + (1 - y) * (1 - z).log()).sum() / 3

    a.backward()
    xut, Wut, yut, zut, aut = x, W, y, z, a

    x = torch.tensor([[1.0, 2.0], 
                      [4.0, 5.0], 
                      [6.0, 7.0]], requires_grad=True, dtype=torch.float64)
    W = torch.tensor([[0.5, -1.0]], requires_grad=True, dtype=torch.float64)
    y = torch.tensor([[1.0], 
                      [1.0], 
                      [0.0]], requires_grad=True, dtype=torch.float64)
    z = (x @ W.T).sigmoid()
    a = -(y * z.log() + (1 - y) * (1 - z).log()).sum() / 3

    a.backward()
    xpt, Wpt, ypt, zpt, apt = x, W, y, z, a

    tol = 1e-6
    # forward
    assert (np.abs(zut.data - zpt.data.numpy()) < tol).all()
    assert (np.abs(aut.data - apt.data.numpy()) < tol).all()

    # backward
    assert (np.abs(xut.grad - xpt.grad.numpy()) < tol).all()
    assert (np.abs(Wut.grad - Wpt.grad.numpy()) < tol).all()
    assert (np.abs(yut.grad - ypt.grad.numpy()) < tol).all()


def test_liear_bias():

    x = ugrad.Tensor([[1.0, 2.0], 
                      [4.0, -5.0], 
                      [6.0, 7.0]], requires_grad=True)
    W = ugrad.Tensor([[0.5, -1.0]], requires_grad=True)
    b = ugrad.Tensor([0.9], requires_grad=True)
    y = ugrad.Tensor([[1.0], 
                      [-1.0], 
                      [0.0]], requires_grad=True)
    z = (x @ W.T + b)
    y_hat = ((z - y)**2).sum() / 3
    
    y_hat.backward()
    xut, Wut, but, yut, zut, y_hatut = x, W, b, y, z, y_hat

    x = torch.tensor([[1.0, 2.0], 
                      [4.0, -5.0], 
                      [6.0, 7.0]], requires_grad=True, dtype=torch.float64)
    W = torch.tensor([[0.5, -1.0]], requires_grad=True, dtype=torch.float64)
    b = torch.tensor([0.9], requires_grad=True, dtype=torch.float64)
    y = torch.tensor([[1.0], 
                      [-1.0], 
                      [0.0]], requires_grad=True, dtype=torch.float64)
    z = x @ W.T + b
    y_hat = ((z - y)**2).sum() / 3

    y_hat.backward()
    xpt, Wpt, bpt, ypt, zpt, y_hatpt = x, W, b, y, z, y_hat

    tol = 1e-6
    # forward
    assert (np.abs(zut.data - zpt.data.numpy()) < tol).all()
    assert (np.abs(y_hatut.data - y_hatpt.data.numpy()) < tol).all()

    # backward
    assert (np.abs(xut.grad - xpt.grad.numpy()) < tol).all()
    assert (np.abs(Wut.grad - Wpt.grad.numpy()) < tol).all()
    assert (np.abs(but.grad - bpt.grad.numpy()) < tol).all()
    assert (np.abs(yut.grad - ypt.grad.numpy()) < tol).all()


def test_liear_mul():

    x = ugrad.Tensor([[1.0, 2.0], 
                      [4.0, -5.0], 
                      [6.0, 7.0]], requires_grad=True)
    W = ugrad.Tensor([[0.5, -1.0]], requires_grad=True)
    b = ugrad.Tensor([0.9], requires_grad=True)
    y = ugrad.Tensor([[1.0], 
                      [-1.0], 
                      [0.0]], requires_grad=True)
    z = b * x @ W.T * b + b
    y_hat = ((z - y)**2).sum() / 3
    
    y_hat.backward()
    xut, Wut, but, yut, zut, y_hatut = x, W, b, y, z, y_hat

    x = torch.tensor([[1.0, 2.0], 
                      [4.0, -5.0], 
                      [6.0, 7.0]], requires_grad=True, dtype=torch.float64)
    W = torch.tensor([[0.5, -1.0]], requires_grad=True, dtype=torch.float64)
    b = torch.tensor([0.9], requires_grad=True, dtype=torch.float64)
    y = torch.tensor([[1.0], 
                      [-1.0], 
                      [0.0]], requires_grad=True, dtype=torch.float64)
    z = b * x @ W.T * b + b
    y_hat = ((z - y)**2).sum() / 3

    y_hat.backward()
    xpt, Wpt, bpt, ypt, zpt, y_hatpt = x, W, b, y, z, y_hat

    tol = 1e-6
    # forward
    assert (np.abs(zut.data - zpt.data.numpy()) < tol).all()
    assert (np.abs(y_hatut.data - y_hatpt.data.numpy()) < tol).all()

    # backward
    assert (np.abs(xut.grad - xpt.grad.numpy()) < tol).all()
    assert (np.abs(Wut.grad - Wpt.grad.numpy()) < tol).all()
    assert (np.abs(but.grad - bpt.grad.numpy()) < tol).all()
    assert (np.abs(yut.grad - ypt.grad.numpy()) < tol).all()

    
def test_log_softmax():
    xx = np.array([[1., 2., 3., 3., 2., 1., 0.], [2.0, 2.5, -3., -3., 2.5, 1.0, 0.]])
    yy = np.array([[0., 0., 1., 1., 0., 0., 0.], [1., 1., 0., 0., 1., 0., 0.]])

    x = ugrad.Tensor(xx, requires_grad=True)
    y = ugrad.Tensor(yy, requires_grad=True)

    y_hat = x.log_softmax(dim=1)
    l = -(y * y_hat).sum()
    l.backward()

    xut, yut, y_hatut, lut = x, y, y_hat, l

    x = torch.tensor(xx, requires_grad=True, dtype=torch.float64)
    y = torch.tensor(yy, requires_grad=True, dtype=torch.float64)

    y_hat = x.log_softmax(dim=1)
    l = -(y * y_hat).sum()
    l.backward()

    xpt, ypt, y_hatpt, lpt = x, y, y_hat, l

    tol = 1e-6
    # forward
    assert (np.abs(lut.data - lpt.data.numpy()) < tol).all()
    assert (np.abs(y_hatut.data - y_hatpt.data.numpy()) < tol).all()

    # backward
    assert (np.abs(xut.grad - xpt.grad.numpy()) < tol).all()
    assert (np.abs(yut.grad - ypt.grad.numpy()) < tol).all()


def test_log_softmax_celoss():
    xx = np.array([[1., 2., 3.5, 3., 2., 1., 0.], [2.0, 2.5, -3., -3., 2.5, 1.0, 0.]])
    yy = np.array([[0., 0., 1., 0., 0., 0., 0.], [0., 0., 0., 0., 1., 0., 0.]])

    x = ugrad.Tensor(xx, requires_grad=True)
    y = ugrad.Tensor(yy, requires_grad=True)

    y_hat = x.log_softmax(dim=1)
    l = -(y * y_hat).sum() / 2
    l.backward()

    xut, yut, lut = x, y, l

    x = torch.tensor(xx, requires_grad=True, dtype=torch.float64)
    y = torch.tensor(yy, requires_grad=True, dtype=torch.float64)

    l = torch.nn.functional.cross_entropy(x, y)
    l.backward()

    xpt, ypt, lpt = x, y, l

    tol = 1e-6
    # forward
    assert (np.abs(lut.data - lpt.data.numpy()) < tol).all()

    # backward
    assert (np.abs(xut.grad - xpt.grad.numpy()) < tol).all()
    assert (np.abs(yut.grad - ypt.grad.numpy()) < tol).all()

