import ugrad
import numpy as np
import torch


def test_matmul_1_1():

    xn = np.array([1.0, 2.0, 3.0])
    yn = np.array([4.0, 5.0, 6.0])
    xu = ugrad.TensorBase([1.0, 2.0, 3.0])
    yu = ugrad.TensorBase([4.0, 5.0, 6.0])

    resn = xn @ yn
    resu = xu @ yu

    assert (xu.numpy() == xn).all
    assert (yu.numpy() == yn).all
    assert xu.shape == xn.shape
    assert yu.shape == yn.shape
    assert resu.numpy() == resn
    assert resu.shape ==resn.shape


def test_matmul_1_2():

    xn = np.array([1.0, 2.0])
    yn = np.array([[3.0, 4.0, 5.0],
                   [6.0, 7.0, 8.0]])
    xu = ugrad.TensorBase([1.0, 2.0])
    yu = ugrad.TensorBase([[3.0, 4.0, 5.0], 
                           [6.0, 7.0, 8.0]])

    resn = xn @ yn
    resu = xu @ yu

    assert (xu.numpy() == xn).all
    assert (yu.numpy() == yn).all
    assert xu.shape == xn.shape
    assert yu.shape == yn.shape
    assert (resu.numpy() == resn).all()
    assert resu.shape ==resn.shape

def test_matmul_1_3():

    xn = np.array([1.0, 2.0, 3.0])                   # (3,) ->                      (1, 3) -> (2, 4, 3) \_ (2, 4, 1) -> (2, 4)
    yn = np.array([[[5.0,   6.0,  7.0,  8.0],        # (2, 3, 4) -> (2, 4, 3) -> (2, 4, 3) -> (2, 4, 3) /
                    [9.0,  10.0, 11.0, 12.0], 
                    [13.0, 14.0, 15.0, 16.0]], 
                   [[17.0, 18.0, 19.0, 20.0],
                    [21.0, 22.0, 23.0, 24.0],
                    [25.0, 26.0, 27.0, 28.0]]])
    xu = ugrad.TensorBase([1.0, 2.0, 3.0])
    yu = ugrad.TensorBase([[[5.0,   6.0,  7.0,  8.0], 
                            [9.0,  10.0, 11.0, 12.0], 
                            [13.0, 14.0, 15.0, 16.0]], 
                           [[17.0, 18.0, 19.0, 20.0],
                            [21.0, 22.0, 23.0, 24.0],
                            [25.0, 26.0, 27.0, 28.0]]])

    resn = xn @ yn
    resu = xu @ yu

    assert (xu.numpy() == xn).all
    assert (yu.numpy() == yn).all
    assert xu.shape == xn.shape
    assert yu.shape == yn.shape
    assert (resu.numpy() == resn).all()
    assert resu.shape ==resn.shape


def test_matmul_2_1():

    xn = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    yn = np.array([7.0, 8.0, 9.0])
    xu = ugrad.TensorBase([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    yu = ugrad.TensorBase([7.0, 8.0, 9.0])

    resn = xn @ yn
    resu = xu @ yu

    assert (xu.numpy() == xn).all
    assert (yu.numpy() == yn).all
    assert xu.shape == xn.shape
    assert yu.shape == yn.shape
    assert (resu.numpy() == resn).all()
    assert resu.shape ==resn.shape

def test_matmul_2_2():

    xn = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])     # (2, 3) ->           (2, 1, 3) -> (2, 2, 3)  \_ (2, 2, 1) -> (2, 2)
    yn = np.array([[7.0, 8.0], [10.0, 11.0], [12., 13.]]) # (3, 2) -> (2, 3) -> (1, 2, 3) -> (2, 2, 3)  /
    xu = ugrad.TensorBase([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    yu = ugrad.TensorBase([[7.0, 8.0], [10.0, 11.0], [12., 13.]])

    resn = xn @ yn
    resu = xu @ yu

    assert (xu.numpy() == xn).all
    assert (yu.numpy() == yn).all
    assert xu.shape == xn.shape
    assert yu.shape == yn.shape
    assert (resu.numpy() == resn).all()
    assert resu.shape ==resn.shape

def test_matmul_2_3():

    xn = np.array([[1.0,   2.0,  3.0],
                   [4.0,   5.0,  6.0],
                   [7.0,   8.0,  9.0],
                   [10.0, 11.0, 12.0],
                   [11.0, 12.0, 13.0]])         # (5, 3) ->    (5, 1, 3) ->    (5, 4, 3)    (2, 5, 4, 3) \_ (2, 5, 4, 1) -> (2, 5, 4)
    yn = np.array([[[5.0,   6.0,  7.0,  8.0],   # (2, 3, 4) -> (2, 4, 3) -> (2, 1, 4, 3) -> (2, 5, 4, 3) /
                    [9.0,  10.0, 11.0, 12.0], 
                    [13.0, 14.0, 15.0, 16.0]], 
                   [[17.0, 18.0, 19.0, 20.0],
                    [21.0, 22.0, 23.0, 24.0],
                    [25.0, 26.0, 27.0, 28.0]]])
    xu = ugrad.TensorBase([[1.0,   2.0,  3.0],
                           [4.0,   5.0,  6.0],
                           [7.0,   8.0,  9.0],
                           [10.0, 11.0, 12.0],
                           [11.0, 12.0, 13.0]],)
    yu = ugrad.TensorBase([[[5.0,   6.0,  7.0,  8.0], 
                            [9.0,  10.0, 11.0, 12.0], 
                            [13.0, 14.0, 15.0, 16.0]], 
                           [[17.0, 18.0, 19.0, 20.0],
                            [21.0, 22.0, 23.0, 24.0],
                            [25.0, 26.0, 27.0, 28.0]]])

    resn = xn @ yn
    resu = xu @ yu

    assert (xu.numpy() == xn).all
    assert (yu.numpy() == yn).all
    assert xu.shape == xn.shape
    assert yu.shape == yn.shape
    assert (resu.numpy() == resn).all()
    assert resu.shape ==resn.shape

def test_matmul_3_1():

    xn = np.array([[[5.0,   6.0,  7.0,  8.0],
                    [9.0,  10.0, 11.0, 12.0], 
                    [13.0, 14.0, 15.0, 16.0]], 
                   [[17.0, 18.0, 19.0, 20.0],
                    [21.0, 22.0, 23.0, 24.0],
                    [25.0, 26.0, 27.0, 28.0]]])         # (2, 3, 4) -> (2, 3, 4) -> (2, 3, 4)  \_ (2, 3, 1) -> (2, 3) 
    yn = np.array([1.0, 2.0, 3.0, 4.0])                 # (4,)      ->    (1, 4) -> (2, 3, 4)  /
    xu = ugrad.TensorBase([[[5.0,   6.0,  7.0,  8.0],
                            [9.0,  10.0, 11.0, 12.0], 
                            [13.0, 14.0, 15.0, 16.0]], 
                           [[17.0, 18.0, 19.0, 20.0],
                            [21.0, 22.0, 23.0, 24.0],
                            [25.0, 26.0, 27.0, 28.0]]])
    yu = ugrad.TensorBase([1.0, 2.0, 3.0, 4.0])

    resn = xn @ yn
    resu = xu @ yu

    assert (xu.numpy() == xn).all
    assert (yu.numpy() == yn).all
    assert xu.shape == xn.shape
    assert yu.shape == yn.shape
    assert (resu.numpy() == resn).all()
    assert resu.shape ==resn.shape

def test_matmul_3_2():

    xn = np.array([[[5.0,   6.0,  7.0,  8.0],
                    [9.0,  10.0, 11.0, 12.0], 
                    [13.0, 14.0, 15.0, 16.0]], 
                   [[17.0, 18.0, 19.0, 20.0],
                    [21.0, 22.0, 23.0, 24.0],
                    [25.0, 26.0, 27.0, 28.0]]])     # (2, 3, 4) -> (2, 3, 4) -> (2, 3, 1, 4) -> (2, 3, 5, 4) \_ (2, 3, 5, 1) -> (2, 3, 5) 
    yn = np.array([[1.0,   2.0,  3.0,  4.0,  5.0],  # (4, 5)    ->    (5, 4) ->    (1, 5, 4) -> (2, 3, 5, 4) /
                   [6.0,   7.0,  8.0,  9.0, 10.0],
                   [11.0, 12.0, 13.0, 14.0, 15.0],
                   [16.0, 17.0, 18.0, 19.0, 20.0]])          
    xu = ugrad.TensorBase([[[5.0,   6.0,  7.0,  8.0],
                            [9.0,  10.0, 11.0, 12.0], 
                            [13.0, 14.0, 15.0, 16.0]], 
                           [[17.0, 18.0, 19.0, 20.0],
                            [21.0, 22.0, 23.0, 24.0],
                            [25.0, 26.0, 27.0, 28.0]]])
    yu = ugrad.TensorBase([[1.0,   2.0,  3.0,  4.0,  5.0],
                           [6.0,   7.0,  8.0,  9.0, 10.0],
                           [11.0, 12.0, 13.0, 14.0, 15.0],
                           [16.0, 17.0, 18.0, 19.0, 20.0]])

    resn = xn @ yn
    resu = xu @ yu

    assert (xu.numpy() == xn).all
    assert (yu.numpy() == yn).all
    assert xu.shape == xn.shape
    assert yu.shape == yn.shape
    assert (resu.numpy() == resn).all()
    assert resu.shape ==resn.shape


def test_matmul_3_3():

    xn = np.array([[[5.0,   6.0,  7.0,  8.0],
                    [9.0,  10.0, 11.0, 12.0], 
                    [13.0, 14.0, 15.0, 16.0]], 
                   [[17.0, 18.0, 19.0, 20.0],
                    [21.0, 22.0, 23.0, 24.0],
                    [25.0, 26.0, 27.0, 28.0]]])     # (2, 3, 4) -> (2, 3, 4) -> (2, 3, 1, 4) -> (2, 3, 5, 4) \_ (2, 3, 5, 1) -> (2, 3, 5) 
    yn = np.array([[[1.0,  2.0,  3.0,  4.0,  5.0],  # (2, 4, 5) -> (2, 5, 4) -> (2, 1, 5, 4) -> (2, 3, 5, 4) /
                   [6.0,   7.0,  8.0,  9.0, 10.0],
                   [11.0, 12.0, 13.0, 14.0, 15.0],
                   [16.0, 17.0, 18.0, 19.0, 20.0]],
                  [[21.0, 22.0, 23.0, 24.0, 25.0],
                   [26.0, 27.0, 28.0, 29.0, 30.0],
                   [31.0, 32.0, 33.0, 34.0, 35.0],
                   [36.0, 37.0, 38.0, 39.0, 40.0]]])          
    xu = ugrad.TensorBase([[[5.0,   6.0,  7.0,  8.0],
                            [9.0,  10.0, 11.0, 12.0], 
                            [13.0, 14.0, 15.0, 16.0]], 
                           [[17.0, 18.0, 19.0, 20.0],
                            [21.0, 22.0, 23.0, 24.0],
                            [25.0, 26.0, 27.0, 28.0]]])
    yu = ugrad.TensorBase([[[1.0,  2.0,  3.0,  4.0,  5.0],
                            [6.0,   7.0,  8.0,  9.0, 10.0],
                            [11.0, 12.0, 13.0, 14.0, 15.0],
                            [16.0, 17.0, 18.0, 19.0, 20.0]],
                           [[21.0, 22.0, 23.0, 24.0, 25.0],
                            [26.0, 27.0, 28.0, 29.0, 30.0],
                            [31.0, 32.0, 33.0, 34.0, 35.0],
                            [36.0, 37.0, 38.0, 39.0, 40.0]]])

    resn = xn @ yn
    resu = xu @ yu

    assert (xu.numpy() == xn).all
    assert (yu.numpy() == yn).all
    assert xu.shape == xn.shape
    assert yu.shape == yn.shape
    assert (resu.numpy() == resn).all()
    assert resu.shape ==resn.shape


def test_expand_dims_1():

    xns = [np.array(1.0),
           np.array([1.0]),
           np.array([1.0, 2.0, 3.0]),
           np.array([[1.0, 2.0, 3.0], [4., 5., 6.]]),
    ]
    xus = [ugrad.TensorBase(1.0),
           ugrad.TensorBase([1.0]),
           ugrad.TensorBase([1.0, 2.0, 3.0]),
           ugrad.TensorBase([[1.0, 2.0, 3.0], [4., 5., 6.]]),
    ]

    dims = [0, (0, 1), (1, 0), (-1, -2), (-2, -1), (-1, 0), (0, -1), 
            (-1, -2, -3), (-1, -3, -2), (-2, -1, -3), (-2, -3, -1), (-3, -2, -1), (-3, -1, -2), 
            (0,  -1, -2),  (0, -2, -1), (-1,  0, -2), (-1, -2,  0), (-2,  0, -1), (-2, -1, 0),
            (0, 1, 2, 3, 4, 5), (-1, -2, -3, -4, -5, -6)]

    for dim in dims:
        for xn, xu in zip(xns, xus):
            resn = np.expand_dims(xn, dim)
            resu = xu.expand_dims(dim)

            assert (xu.numpy() == xn).all
            assert xu.shape == xn.shape
            assert (resu.numpy() == resn).all()
            assert resu.shape ==resn.shape

def test_expand_dims_2():

    xns = [np.array([[1.0, 2.0, 3.0], [4., 5., 6.]]),
           np.array([[[5.0,   6.0,  7.0,  8.0],
                      [9.0,  10.0, 11.0, 12.0], 
                      [13.0, 14.0, 15.0, 16.0]], 
                     [[17.0, 18.0, 19.0, 20.0],
                      [21.0, 22.0, 23.0, 24.0],
                      [25.0, 26.0, 27.0, 28.0]]])
    ]
    xus = [ugrad.TensorBase([[1.0, 2.0, 3.0], [4., 5., 6.]]),
           ugrad.TensorBase([[[5.0,   6.0,  7.0,  8.0],
                              [9.0,  10.0, 11.0, 12.0], 
                              [13.0, 14.0, 15.0, 16.0]], 
                             [[17.0, 18.0, 19.0, 20.0],
                              [21.0, 22.0, 23.0, 24.0],
                              [25.0, 26.0, 27.0, 28.0]]])
    ]

    dims = [0, 1, 2, (1, 2), (2, 1), (2, 3), (3, 2), (0, 2), (0, -2), 
            (1, 2, 3, 4, 5), (-2, -3, -4, -5, -6)]

    for dim in dims:
        for xn, xu in zip(xns, xus):
            resn = np.expand_dims(xn, dim)
            resu = xu.expand_dims(dim)

            assert (xu.numpy() == xn).all
            assert xu.shape == xn.shape
            assert (resu.numpy() == resn).all()
            assert resu.shape ==resn.shape

def test_init_list_np():
    xn = np.array([[[5.0,   6.0,  7.0,  8.0],
                    [9.0,  10.0, 11.0, 12.0], 
                    [13.0, 14.0, 15.0, 16.0]], 
                   [[17.0, 18.0, 19.0, 20.0],
                    [21.0, 22.0, 23.0, 24.0],
                    [25.0, 26.0, 27.0, 28.0]]])
    
    xu = ugrad.TensorBase([[[5.0,   6.0,  7.0,  8.0],
                            [9.0,  10.0, 11.0, 12.0], 
                            [13.0, 14.0, 15.0, 16.0]], 
                           [[17.0, 18.0, 19.0, 20.0],
                            [21.0, 22.0, 23.0, 24.0],
                            [25.0, 26.0, 27.0, 28.0]]])
    
    xun = ugrad.TensorBase(xn)

    xu_numpy = xu.numpy()
    xun_numpy = xun.numpy()
    assert (xu_numpy == xn).all
    assert (xun_numpy == xn).all
    assert (xu_numpy.shape == xn.shape)
    assert (xun_numpy.shape == xn.shape)
    assert (xu_numpy.dtype == xn.dtype)
    assert (xun_numpy.dtype == xn.dtype)
    assert (xn.shape == (2, 3, 4))
    assert (xn.dtype == np.float64)
    assert (xu.dtype == "double")
    assert (xun.dtype == "double")


    xn = np.array([[[5,   6,  7,  8],
                    [9,  10, 11, 12], 
                    [13, 14, 15, 16]], 
                   [[17, 18, 19, 20],
                    [21, 22, 23, 24],
                    [25, 26, 27, 28]]])
    
    xu = ugrad.TensorBase([[[5,   6,  7,  8],
                            [9,  10, 11, 12], 
                            [13, 14, 15, 16]], 
                           [[17, 18, 19, 20],
                            [21, 22, 23, 24],
                            [25, 26, 27, 28]]])
    
    xun = ugrad.TensorBase(xn)

    xu_numpy = xu.numpy()
    xun_numpy = xun.numpy()
    assert (xu_numpy == xn).all
    assert (xun_numpy == xn).all
    assert (xu_numpy.shape == xn.shape)
    assert (xun_numpy.shape == xn.shape)
    assert (xu_numpy.dtype == xn.dtype)
    assert (xun_numpy.dtype == xn.dtype)
    assert (xn.shape == (2, 3, 4))
    assert (xn.dtype == np.int64)
    assert (xu.dtype == "long")
    assert (xun.dtype == "long")

def test_broadcast_to():
    tests = [(np.arange(1).reshape((1, )), (2, 3)),
             (np.arange(1).reshape((1, )), (1, )),
             (np.arange(1).reshape((1, )), (1, 1)),
             (np.arange(2 * 3 * 4).reshape((2, 1, 12)), (2, 5, 12)),
             (np.arange(2 * 3 * 4).reshape((2, 1, 1, 12)), (2, 5, 5, 12)),
             (np.arange(2 * 3 * 4).reshape((2, 12)), (1, 2, 12)),
             (np.arange(2 * 3 * 4).reshape((2, 3, 4)), (5, 2, 3, 4)),
             (np.arange(2 * 3 * 4).reshape((2, 3, 4)), (1, 5, 2, 3, 4)),
             (np.arange(2 * 3 * 4).reshape((2, 3, 4)), (5, 1, 2, 3, 4)),
             (np.arange(2 * 3 * 4).reshape((2, 3, 4)), (5, 1, 1, 1, 2, 3, 4)),
             (np.arange(2 * 3 * 4).reshape((2, 3, 4)), (1, 1, 1, 5, 2, 3, 4)),
             (np.arange(2 * 3 * 4).reshape((2, 3, 1, 4)), (1, 1, 1, 5, 2, 3, 7, 4)),
             (np.arange(2 * 3 * 4).reshape((2, 3, 1, 1, 4)), (1, 1, 1, 5, 2, 3, 1, 7, 4)),
             (np.arange(2 * 3 * 4).reshape((2, 3, 1, 1, 4)), (1, 1, 1, 5, 2, 3, 7, 1, 4)),
             (np.arange(2 * 3 * 4).reshape((1, 2, 3, 1, 1, 4)), (1, 2, 3, 1, 1, 4)),
             (np.arange(2 * 3 * 4).reshape((2, 1, 3, 4)), (1, 1, 1, 5, 2, 7, 3, 4)),
             (np.arange(2 * 3 * 4).reshape((2, 1, 1, 3, 4)), (1, 1, 1, 5, 2, 1, 7, 3, 4)),
             (np.arange(2 * 3 * 4).reshape((2, 1, 1, 3, 4)), (1, 1, 1, 5, 2, 7, 1, 3, 4)),
            ]

    for xn, bcast_shape in tests:
        xn_b = np.broadcast_to(xn, bcast_shape)
        xu_b = ugrad.TensorBase(xn).broadcast_to(bcast_shape)
        assert (xu_b.numpy() == xn_b).all
        assert (xu_b.shape == xn_b.shape)
        assert (xu_b.strides == tuple(stride // xn.itemsize for stride in xn_b.strides))


def test_broadcast_to_T():
    tests = [(np.arange(2 * 3 * 4).reshape((2, 1, 12)), (2, 12, 5)),
             (np.arange(2 * 3 * 4).reshape((2, 1, 1, 12)), (2, 5, 12, 5)),
             (np.arange(2 * 3 * 4).reshape((2, 12)), (1, 12, 2)),
             (np.arange(2 * 3 * 4).reshape((2, 3, 4)), (5, 2, 4, 3)),
             (np.arange(2 * 3 * 4).reshape((2, 3, 4)), (1, 5, 2, 4, 3)),
             (np.arange(2 * 3 * 4).reshape((2, 3, 4)), (5, 1, 2, 4, 3)),
             (np.arange(2 * 3 * 4).reshape((2, 3, 4)), (5, 1, 1, 1, 2, 4, 3)),
             (np.arange(2 * 3 * 4).reshape((2, 3, 4)), (1, 1, 1, 5, 2, 4, 3)),
             (np.arange(2 * 3 * 4).reshape((2, 3, 1, 4)), (1, 1, 1, 5, 2, 3, 4, 7)),
             (np.arange(2 * 3 * 4).reshape((2, 3, 1, 1, 4)), (1, 1, 1, 5, 2, 3, 1, 4, 7)),
             (np.arange(2 * 3 * 4).reshape((2, 3, 1, 1, 4)), (1, 1, 1, 5, 2, 3, 7, 4, 1)),
             (np.arange(2 * 3 * 4).reshape((1, 2, 3, 1, 1, 4)), (1, 2, 3, 1, 4, 1)),
             (np.arange(2 * 3 * 4).reshape((2, 1, 3, 4)), (1, 1, 1, 5, 2, 7, 4, 3)),
             (np.arange(2 * 3 * 4).reshape((2, 1, 1, 3, 4)), (1, 1, 1, 5, 2, 1, 7, 4, 3)),
             (np.arange(2 * 3 * 4).reshape((2, 1, 1, 3, 4)), (1, 1, 1, 5, 2, 7, 1, 4, 3)),
            ]

    for xn, bcast_shape in tests:
        xn_b = np.broadcast_to(xn.swapaxes(-1, -2), bcast_shape)
        xu_b = ugrad.TensorBase(xn).T.broadcast_to(bcast_shape)
        assert (xu_b.numpy() == xn_b).all
        assert (xu_b.shape == xn_b.shape)
        assert (xu_b.strides == tuple(stride // xn.itemsize for stride in xn_b.strides))


def test_broadcast_to_torch():
    tests = [(np.arange(1).reshape((1, )), (2, 3)),
             (np.arange(1).reshape((1, )), (1, )),
             (np.arange(1).reshape((1, )), (1, 1)),
             (np.arange(2 * 3 * 4).reshape((2, 1, 12)), (2, 5, 12)),
             (np.arange(2 * 3 * 4).reshape((2, 1, 1, 12)), (2, 5, 5, 12)),
             (np.arange(2 * 3 * 4).reshape((2, 12)), (1, 2, 12)),
             (np.arange(2 * 3 * 4).reshape((2, 3, 4)), (5, 2, 3, 4)),
             (np.arange(2 * 3 * 4).reshape((2, 3, 4)), (1, 5, 2, 3, 4)),
             (np.arange(2 * 3 * 4).reshape((2, 3, 4)), (5, 1, 2, 3, 4)),
             (np.arange(2 * 3 * 4).reshape((2, 3, 4)), (5, 1, 1, 1, 2, 3, 4)),
             (np.arange(2 * 3 * 4).reshape((2, 3, 4)), (1, 1, 1, 5, 2, 3, 4)),
             (np.arange(2 * 3 * 4).reshape((2, 3, 1, 4)), (1, 1, 1, 5, 2, 3, 7, 4)),
             (np.arange(2 * 3 * 4).reshape((2, 3, 1, 1, 4)), (1, 1, 1, 5, 2, 3, 1, 7, 4)),
             (np.arange(2 * 3 * 4).reshape((2, 3, 1, 1, 4)), (1, 1, 1, 5, 2, 3, 7, 1, 4)),
             (np.arange(2 * 3 * 4).reshape((1, 2, 3, 1, 1, 4)), (1, 2, 3, 1, 1, 4)),
             (np.arange(2 * 3 * 4).reshape((2, 1, 3, 4)), (1, 1, 1, 5, 2, 7, 3, 4)),
             (np.arange(2 * 3 * 4).reshape((2, 1, 1, 3, 4)), (1, 1, 1, 5, 2, 1, 7, 3, 4)),
             (np.arange(2 * 3 * 4).reshape((2, 1, 1, 3, 4)), (1, 1, 1, 5, 2, 7, 1, 3, 4)),
            ]

    for xn, bcast_shape in tests:
        xt_b = torch.tensor(xn).broadcast_to(bcast_shape)
        xu_b = ugrad.TensorBase(xn).broadcast_to_torch(bcast_shape)
        assert (xu_b.numpy() == xt_b.numpy()).all
        assert (xu_b.shape == tuple(xt_b.shape))
        assert (xu_b.strides == xt_b.stride())
    #assert False

def test_broadcast_to_torch_T():
    tests = [(np.arange(2 * 3 * 4).reshape((2, 1, 12)), (2, 12, 5)),
             (np.arange(2 * 3 * 4).reshape((2, 1, 1, 12)), (2, 5, 12, 5)),
             (np.arange(2 * 3 * 4).reshape((2, 12)), (1, 12, 2)),
             (np.arange(2 * 3 * 4).reshape((2, 3, 4)), (5, 2, 4, 3)),
             (np.arange(2 * 3 * 4).reshape((2, 3, 4)), (1, 5, 2, 4, 3)),
             (np.arange(2 * 3 * 4).reshape((2, 3, 4)), (5, 1, 2, 4, 3)),
             (np.arange(2 * 3 * 4).reshape((2, 3, 4)), (5, 1, 1, 1, 2, 4, 3)),
             (np.arange(2 * 3 * 4).reshape((2, 3, 4)), (1, 1, 1, 5, 2, 4, 3)),
             (np.arange(2 * 3 * 4).reshape((2, 3, 1, 4)), (1, 1, 1, 5, 2, 3, 4, 7)),
             (np.arange(2 * 3 * 4).reshape((2, 3, 1, 1, 4)), (1, 1, 1, 5, 2, 3, 1, 4, 7)),
             (np.arange(2 * 3 * 4).reshape((2, 3, 1, 1, 4)), (1, 1, 1, 5, 2, 3, 7, 4, 1)),
             (np.arange(2 * 3 * 4).reshape((1, 2, 3, 1, 1, 4)), (1, 2, 3, 1, 4, 1)),
             (np.arange(2 * 3 * 4).reshape((2, 1, 3, 4)), (1, 1, 1, 5, 2, 7, 4, 3)),
             (np.arange(2 * 3 * 4).reshape((2, 1, 1, 3, 4)), (1, 1, 1, 5, 2, 1, 7, 4, 3)),
             (np.arange(2 * 3 * 4).reshape((2, 1, 1, 3, 4)), (1, 1, 1, 5, 2, 7, 1, 4, 3)),
            ]

    for xn, bcast_shape in tests:
        xt_b = torch.tensor(xn).mT.broadcast_to(bcast_shape)
        xu_b = ugrad.TensorBase(xn).T.broadcast_to_torch(bcast_shape)
        #print(xu_b.strides, xt_b.stride())
        assert (xu_b.numpy() == xt_b.numpy()).all
        assert (xu_b.shape == tuple(xt_b.shape))
        assert (xu_b.strides == xt_b.stride())
