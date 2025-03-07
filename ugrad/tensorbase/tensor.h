#ifndef TENSOR_H
#define TENSOR_H

#define DOUBLE_DTYPE 0
#define LONG_DTYPE 1
#define DTYPES(dtype1, dtype2) (dtype1 | (dtype2 << 2))

#define CPU_DEVICE -1

extern "C"
typedef struct {
    int nbytes;
    int nshares;
    int device;
    void *data;
} Storage;

extern "C"
typedef struct {
    Storage *storage;
    int *shape;
    int *strides;
    int offset;
    int ndim;
    int numel;
    int dtype;
    int device;
} Tensor;

Storage *create_storage(int nbytes, int device);
Storage *clone_storage(Storage *s);
extern "C" Storage *cc_storage(int nbytes, void *data);
extern "C" Tensor *create_tensor(Storage *storage, int *shape, int ndim, int dtype);
extern "C" void delete_tensor(Tensor *t);
extern "C" void update_tensor(Tensor *t, int *shape, int *strides, int ndim);
extern "C" Tensor *clone_tensor(Tensor *t);
extern "C" Tensor *to(Tensor *t, int device);
extern "C" void *get_item(Tensor *t, int idx);
extern "C" bool is_contiguous(Tensor *t);
Tensor *sum_along_axis(Tensor *t, int axis);
extern "C" Tensor *sum(Tensor *t, int *axis, int axis_size);
Tensor *max_along_axis(Tensor *t, int axis);
extern "C" Tensor *max_t(Tensor *t, int *axis, int axis_size);
extern "C" Tensor *argmax(Tensor *t, int axis);
extern "C" Tensor *gt(Tensor *t, Tensor *t2);
extern "C" Tensor *eq(Tensor *t, Tensor *t2);
extern "C" Tensor *add(Tensor *t, Tensor *t2);
extern "C" Tensor *mul(Tensor *t, Tensor *t2);
extern "C" Tensor *maximum(Tensor *t, Tensor *t2);
extern "C" Tensor *mul_reduce(Tensor *t, Tensor *t2, int axis);
extern "C" Tensor *pow_t(Tensor *t, double x);
extern "C" Tensor *exp_t(Tensor *t);
extern "C" Tensor *log_t(Tensor *t);
extern "C" Tensor *tanh_t(Tensor *t);
extern "C" Tensor *contiguous(Tensor *t);

#endif
