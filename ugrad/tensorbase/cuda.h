#ifndef CUDA_H
#define CUDA_H
#include "tensor.h"

#define THREADS_PER_BLOCK 128

typedef struct {
    int *shape;
    int *strides;
    int offset;
    int ndim;
    int numel;
} TensorParams;

Tensor *to_cuda(Tensor *t, int device);
Tensor *to_cpu(Tensor *t);
void cuda_free(void *ptr);
Storage *create_cuda_storage(int nbytes, int device);
Storage *clone_cuda_storage(Storage *s);
void *get_item_cuda(Tensor *t, int idx);
Tensor *sum_cuda(Tensor *t, int axis);
Tensor *max_cuda(Tensor *t, int axis);
Tensor *argmax_cuda(Tensor *t, int axis);
Tensor *gt_cuda(Tensor *t, Tensor *t2);
Tensor *eq_cuda(Tensor *t, Tensor *t2);
Tensor *add_cuda(Tensor *t, Tensor *t2);
Tensor *mul_cuda(Tensor *t, Tensor *t2);
void assign_cuda(Tensor *t, Tensor *t2);
void add_at_cuda(Tensor *t, Tensor *idx, Tensor *t2);
void uniform_cuda(Tensor *t, double a, double b);
Tensor *maximum_cuda(Tensor *t, Tensor *t2);
Tensor *mul_reduce_cuda(Tensor *t, Tensor *t2, int axis);
Tensor *pow_cuda(Tensor *t, double x);
Tensor *exp_cuda(Tensor *t);
Tensor *log_cuda(Tensor *t);
Tensor *tanh_cuda(Tensor *t);
Tensor *contiguous_cuda(Tensor *t);
Tensor *arange_cuda(Tensor *t, int start, int stop, int step);

#endif