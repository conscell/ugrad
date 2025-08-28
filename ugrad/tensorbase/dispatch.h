#include "tensor.h"


constexpr int type_code(double) { return DOUBLE_DTYPE; }
constexpr int type_code(long) { return LONG_DTYPE; }


Tensor *dispatch(Tensor *t, auto op, auto func, auto...args) {
    switch (t->dtype) {
        case DOUBLE_DTYPE:
            return op((double *) t->storage->data, t, func, args...);
        case LONG_DTYPE:
            return op((long *) t->storage->data, t, func, args...);
        default:
            return NULL;
    }
}

Tensor *dispatch(Tensor *t, Tensor *t2, auto op, auto func, auto...args) {
    switch(DTYPES(t->dtype, t2->dtype)) {
        case DTYPES(DOUBLE_DTYPE, DOUBLE_DTYPE):
            return op((double *) t->storage->data, (double *) t2->storage->data, t, t2, func, args...);
        case DTYPES(DOUBLE_DTYPE, LONG_DTYPE):
            return op((double *) t->storage->data, (long *) t2->storage->data, t, t2, func, args...);
        case DTYPES(LONG_DTYPE, DOUBLE_DTYPE):
            return op((long *) t->storage->data, (double *) t2->storage->data, t, t2, func, args...);
        case DTYPES(LONG_DTYPE, LONG_DTYPE):
            return op((long *) t->storage->data, (long *) t2->storage->data, t, t2, func, args...);
        default:
            return NULL;
    }
}

Tensor *dispatch_unary_reduce_op(auto *tptr, Tensor *t, auto func, int axis){
    int numel = t->numel / t->shape[axis];
    int *shape = (int *) malloc(t->ndim * sizeof(int));
    if (shape == NULL && t->ndim) {
        perror("malloc");
        exit(1);
    }
    memcpy(shape, t->shape, t->ndim * sizeof(int));
    shape[axis] = 1;

    using restype = std::remove_pointer_t<decltype(tptr)>;
    Tensor *res = create_tensor(create_storage(numel * sizeof(restype), t->device), shape, t->ndim, type_code(restype{}));
    free(shape);

    auto resptr = (restype *) res->storage->data;

    func(tptr, resptr, axis, t, res);

    return res;
}

Tensor *dispatch_argmax(auto *tptr, Tensor *t, auto func, int axis){
    int numel = t->numel / t->shape[axis];
    int *shape = (int *) malloc(t->ndim * sizeof(int));
    if (shape == NULL && t->ndim) {
        perror("malloc");
        exit(1);
    }
    memcpy(shape, t->shape, t->ndim * sizeof(int));
    shape[axis] = 1;

    using max_ttype = std::remove_pointer_t<decltype(tptr)>;
    Tensor *res = create_tensor(create_storage(numel * sizeof(long), t->device), shape, t->ndim, type_code(long{}));
    Tensor *max_t = create_tensor(create_storage(numel * sizeof(max_ttype), t->device), shape, t->ndim, type_code(max_ttype{}));
    free(shape);

    auto *resptr = (long *) res->storage->data;
    auto *max_tptr = (max_ttype *) max_t->storage->data;

    func(tptr, resptr, max_tptr, axis, t, res);

    delete_tensor(max_t);

    return res;
}

Tensor *dispatch_unary_op(auto*tptr, Tensor *t, auto func){
    using restype = std::remove_pointer_t<decltype(tptr)>;
    Tensor *res = create_tensor(create_storage(t->numel * sizeof(restype), t->device), t->shape, t->ndim, type_code(restype{}));
    auto *resptr = (restype *) res->storage->data;

    func(tptr, resptr, t, res->numel);

    return res;
}

Tensor *dispatch_unary_op_d(auto *tptr, Tensor *t, auto func){
    Tensor *res = create_tensor(create_storage(t->numel * sizeof(double), t->device), t->shape, t->ndim, type_code(double{}));
    auto *resptr = (double *) res->storage->data;

    func(tptr, resptr, t, res->numel);

    return res;
}

Tensor *dispatch_unary_op_d_xtra(auto *tptr, Tensor *t, auto func, double xtra){
    Tensor *res = create_tensor(create_storage(t->numel * sizeof(double), t->device), t->shape, t->ndim, type_code(double{}));
    auto *resptr = (double *) res->storage->data;

    func(tptr, resptr, xtra, t, res->numel);

    return res;
}

Tensor *dispatch_binary_reduce_op(auto *tptr, auto *t2ptr, Tensor *t, Tensor *t2, auto func, int axis) {
    int numel = t->numel / t->shape[axis];
    int *shape = (int *) malloc(t->ndim * sizeof(int));
    if (shape == NULL && t->ndim) {
        perror("malloc");
        exit(1);
    }
    memcpy(shape, t->shape, t->ndim * sizeof(int));
    shape[axis] = 1;
    
    using restype = std::common_type_t<std::remove_pointer_t<decltype(tptr)>, std::remove_pointer_t<decltype(t2ptr)>>;
    Tensor *res = create_tensor(create_storage(numel * sizeof(restype), t->device), shape, t->ndim, type_code(restype{}));
    free(shape);
    auto *resptr = (restype *) res->storage->data;

    func(tptr, t2ptr, resptr, axis, t, t2, res);

    return res;
}

Tensor *dispatch_binary_op(auto *tptr, auto *t2ptr, Tensor *t, Tensor *t2, auto func) {
    using restype = std::common_type_t<std::remove_pointer_t<decltype(tptr)>, std::remove_pointer_t<decltype(t2ptr)>>;
    Tensor *res = create_tensor(create_storage(t->numel * sizeof(restype), t->device), t->shape, t->ndim, type_code(restype{}));
    auto *resptr = (restype *) res->storage->data;

    func(tptr, t2ptr, resptr, t, t2, res->numel);

    return res;
}

Tensor *dispatch_binary_op_l(auto *tptr, auto *t2ptr, Tensor *t, Tensor *t2, auto func) {
    Tensor *res = create_tensor(create_storage(t->numel * sizeof(long), t->device), t->shape, t->ndim, type_code(long{}));
    auto *resptr = (long *) res->storage->data;

    func(tptr, t2ptr, resptr, t, t2, res->numel);

    return res;
}
