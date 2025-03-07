#ifndef CPU_H
#define CPU_H
#include "tensor.h"

Storage *create_cpu_storage(int nbytes);
Storage *clone_cpu_storage(Storage *s);
int storage_idx(Tensor *t, int idx);
void *get_item_cpu(Tensor *t, int idx);
Tensor *sum_cpu(Tensor *t, int axis);
Tensor *max_cpu(Tensor *t, int axis);
Tensor *argmax_cpu(Tensor *t, int axis);
Tensor *gt_cpu(Tensor *t, Tensor *t2);
Tensor *eq_cpu(Tensor *t, Tensor *t2);
Tensor *add_cpu(Tensor *t, Tensor *t2);
Tensor *mul_cpu(Tensor *t, Tensor *t2);
Tensor *maximum_cpu(Tensor *t, Tensor *t2);
Tensor *mul_reduce_cpu(Tensor *t, Tensor *t2, int axis);
Tensor *pow_cpu(Tensor *t, double x);
Tensor *exp_cpu(Tensor *t);
Tensor *log_cpu(Tensor *t);
Tensor *tanh_cpu(Tensor *t);
Tensor *contiguous_cpu(Tensor *t);

#endif