#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "cpu.h"
#include "dispatch.h"


Storage *create_cpu_storage(int nbytes){
    Storage *storage = (Storage *) malloc(sizeof(Storage));
    if (storage == NULL) {
        perror("malloc");
        exit(1);
    }
    storage->nbytes = nbytes;
    storage->nshares = -1;
    storage->device = CPU_DEVICE;
    storage->data = malloc(storage->nbytes);
    if (storage->data == NULL) {
        perror("malloc");
        exit(1);
    }
    return storage;
}

Storage *clone_cpu_storage(Storage *s){
    Storage *storage = create_cpu_storage(s->nbytes);
    memcpy(storage->data, s->data, storage->nbytes);
    return storage;
}

int storage_idx(Tensor *t, int logical_idx){
    int idx = 0;
    int logical_stride = t->numel;

    for (int i = 0; i < t->ndim; i++){
        logical_stride /= t->shape[i];
        idx += (logical_idx / logical_stride) * t->strides[i];
        logical_idx %= logical_stride;
    }
    return idx + t->offset;
}

void *get_item_cpu(Tensor *t, int idx){
    void *data = NULL;
    switch(t->dtype) {
        case DOUBLE_DTYPE: {
            data = (void *) ((double *) t->storage->data + idx);
            break;
        }
        case LONG_DTYPE: {
            data = (void *) ((long *) t->storage->data + idx);
            break;
        }

    }
    return data;
}


void sum_kernel_cpu(auto *tptr, auto *resptr, int axis, Tensor *t, Tensor *res, int i){
    int t_idx;
    int idx;
    resptr[i] = 0;
    for(int j = 0; j < t->shape[axis]; j++){
        t_idx = 0;
        idx = i;
        for(int k = 0; k < t->ndim; k++){
            t_idx += (idx / res->strides[k] + (k == axis ? j : 0)) * t->strides[k];
            idx %= res->strides[k];
        }
        t_idx += t->offset;
        resptr[i] += tptr[t_idx];
    }
}

void launch_sum_kernel_cpu(auto *tptr, auto *resptr, int axis, Tensor *t, Tensor *res){
    #pragma omp parallel for
    for (int i=0; i < res->numel; i++)
        sum_kernel_cpu(tptr, resptr, axis, t, res, i);
}

Tensor *sum_cpu(Tensor *t, int axis) {
    return dispatch(t,
        [](auto... args){return dispatch_unary_reduce_op(args...);}, 
        [](auto... args){launch_sum_kernel_cpu(args...);},
        axis
    );
}


void max_kernel_cpu(double *tptr, double *resptr, int axis, Tensor *t, Tensor *res, int i){
    int t_idx;
    int idx;
    resptr[i] = -INFINITY;
    for(int j = 0; j < t->shape[axis]; j++){
        t_idx = 0;
        idx = i;
        for(int k = 0; k < t->ndim; k++){
            t_idx += (idx / res->strides[k] + (k == axis ? j : 0)) * t->strides[k];
            idx %= res->strides[k];
        }
        t_idx += t->offset;
        if(tptr[t_idx] > resptr[i])
            resptr[i] = tptr[t_idx];
    }
}

void max_kernel_cpu(long *tptr, long *resptr, int axis, Tensor *t, Tensor *res, int i){
    int t_idx;
    int idx;
    resptr[i] = -__LONG_MAX__-1;
    for(int j = 0; j < t->shape[axis]; j++){
        t_idx = 0;
        idx = i;
        for(int k = 0; k < t->ndim; k++){
            t_idx += (idx / res->strides[k] + (k == axis ? j : 0)) * t->strides[k];
            idx %= res->strides[k];
        }
        t_idx += t->offset;
        if(tptr[t_idx] > resptr[i])
            resptr[i] = tptr[t_idx];
    }
}

void launch_max_kernel_cpu(auto *tptr, auto *resptr, int axis, Tensor *t, Tensor *res){
    #pragma omp parallel for
    for (int i=0; i < res->numel; i++)
        max_kernel_cpu(tptr, resptr, axis, t, res, i);
}

Tensor *max_cpu(Tensor *t, int axis) {
    return dispatch(t,
        [](auto... args){return dispatch_unary_reduce_op(args...);},
        [](auto... args){launch_max_kernel_cpu(args...);},
        axis
    );
}


void argmax_kernel_cpu(double *tptr, long *resptr, double *max_tptr, int axis, Tensor *t, Tensor *res, int i){
    int t_idx;
    int idx;
    max_tptr[i] = -INFINITY;
    resptr[i] = 0;
    for(int j = 0; j < t->shape[axis]; j++){
        t_idx = 0;
        idx = i;
        for(int k = 0; k < t->ndim; k++){
            t_idx += (idx / res->strides[k] + (k == axis ? j : 0)) * t->strides[k];
            idx %= res->strides[k];
        }
        t_idx += t->offset;
        if(tptr[t_idx] > max_tptr[i]){
            max_tptr[i] = tptr[t_idx];
            resptr[i] = j;
        }
    }
}

void argmax_kernel_cpu(long *tptr, long *resptr, long *max_tptr, int axis, Tensor *t, Tensor *res, int i){
    int t_idx;
    int idx;
    max_tptr[i] = -__LONG_MAX__-1;
    resptr[i] = 0;
    for(int j = 0; j < t->shape[axis]; j++){
        t_idx = 0;
        idx = i;
        for(int k = 0; k < t->ndim; k++){
            t_idx += (idx / res->strides[k] + (k == axis ? j : 0)) * t->strides[k];
            idx %= res->strides[k];
        }
        t_idx += t->offset;
        if(tptr[t_idx] > max_tptr[i]){
            max_tptr[i] = tptr[t_idx];
            resptr[i] = j;
        }
    }
}

void launch_argmax_kernel_cpu(auto *tptr, long *resptr, auto *max_tptr, int axis, Tensor *t, Tensor *res){
    #pragma omp parallel for
    for (int i=0; i < res->numel; i++)
        argmax_kernel_cpu(tptr, resptr, max_tptr, axis, t, res, i);
}

Tensor *argmax_cpu(Tensor *t, int axis){
    return dispatch(t,
        [](auto... args){return dispatch_argmax(args...);},
        [](auto... args){launch_argmax_kernel_cpu(args...);},
        axis
    );
}


void gt_kernel_cpu(auto *tptr, auto *t2ptr, long *resptr, Tensor *t, Tensor *t2, int i) {
    resptr[i] = tptr[storage_idx(t, i)] > t2ptr[storage_idx(t2, i)] ? 1 : 0;
}

void launch_gt_kernel_cpu(auto *tptr, auto *t2ptr, auto *resptr, Tensor *t, Tensor *t2, int numel) {
    #pragma omp parallel for
    for (int i=0; i < numel; i++)
        gt_kernel_cpu(tptr, t2ptr, resptr, t, t2, i);
}

Tensor *gt_cpu(Tensor *t, Tensor *t2) {
    return dispatch(t, t2,
        [](auto... args){return dispatch_binary_op_l(args...);}, 
        [](auto... args){launch_gt_kernel_cpu(args...);}
    );
}


void eq_kernel_cpu(auto *tptr, auto *t2ptr, long *resptr, Tensor *t, Tensor *t2, int i) {
    resptr[i] = tptr[storage_idx(t, i)] == t2ptr[storage_idx(t2, i)] ? 1 : 0;
}

void launch_eq_kernel_cpu(auto *tptr, auto *t2ptr, auto *resptr, Tensor *t, Tensor *t2, int numel) {
    #pragma omp parallel for
    for (int i=0; i < numel; i++)
        eq_kernel_cpu(tptr, t2ptr, resptr, t, t2, i);
}

Tensor *eq_cpu(Tensor *t, Tensor *t2) {
    return dispatch(t, t2,
        [](auto... args){return dispatch_binary_op_l(args...);}, 
        [](auto... args){launch_eq_kernel_cpu(args...);}
    );
}


void add_kernel_cpu(auto *tptr, auto *t2ptr, auto *resptr, Tensor *t, Tensor *t2, int i) {
    resptr[i] = tptr[storage_idx(t, i)] + t2ptr[storage_idx(t2, i)];
}

void launch_add_kernel_cpu(auto *tptr, auto *t2ptr, auto *resptr, Tensor *t, Tensor *t2, int numel) {
    #pragma omp parallel for
    for (int i=0; i < numel; i++)
        add_kernel_cpu(tptr, t2ptr, resptr, t, t2, i);
}

Tensor *add_cpu(Tensor *t, Tensor *t2) {
    return dispatch(t, t2,
        [](auto... args){return dispatch_binary_op(args...);}, 
        [](auto... args){launch_add_kernel_cpu(args...);}
    );
}


void mul_kernel_cpu(auto *tptr, auto *t2ptr, auto *resptr, Tensor *t, Tensor *t2, int i) {
    resptr[i] = tptr[storage_idx(t, i)] * t2ptr[storage_idx(t2, i)];
}

void launch_mul_kernel_cpu(auto *tptr, auto *t2ptr, auto *resptr, Tensor *t, Tensor *t2, int numel) {
    #pragma omp parallel for
    for (int i=0; i < numel; i++)
        mul_kernel_cpu(tptr, t2ptr, resptr, t, t2, i);
}

Tensor *mul_cpu(Tensor *t, Tensor *t2) {
    return dispatch(t, t2,
        [](auto... args){return dispatch_binary_op(args...);}, 
        [](auto... args){launch_mul_kernel_cpu(args...);}
    );
}


void assign_kernel_cpu(auto *tptr, auto *t2ptr, Tensor *t, Tensor *t2, int i) {
    tptr[storage_idx(t, i)] = t2ptr[storage_idx(t2, i)];
}

void launch_assign_kernel_cpu(auto *tptr, auto *t2ptr, Tensor *t, Tensor *t2, int numel) {
    #pragma omp parallel for
    for (int i=0; i < numel; i++)
        assign_kernel_cpu(tptr, t2ptr, t, t2, i);
}

void assign_cpu(Tensor *t, Tensor *t2) {
    dispatch(t, t2,
        [](auto... args){return dispatch_binary_inplace_op(args...);}, 
        [](auto... args){launch_assign_kernel_cpu(args...);}
    );
}


void add_at_kernel_cpu(auto *tptr, auto *idxptr, auto *t2ptr, Tensor *t, Tensor *idx, Tensor *t2, int i) {
    #pragma omp atomic update
    tptr[storage_idx(t, idxptr[storage_idx(idx, i)])] += t2ptr[storage_idx(t2, i)];
}

void launch_add_at_kernel_cpu(auto *tptr, auto *t2ptr, Tensor *t, Tensor *t2, int numel, Tensor *idx) {
    auto *idxptr = (long *) idx->storage->data;
    auto numel_idx = idx->numel;
    #pragma omp parallel for
    for (int i=0; i < numel_idx; i++)
        add_at_kernel_cpu(tptr, idxptr, t2ptr, t, idx, t2, i);
}

void add_at_cpu(Tensor *t, Tensor *idx, Tensor *t2){
    dispatch(t, t2,
        [](auto... args){return dispatch_binary_inplace_op(args...);}, 
        [](auto... args){launch_add_at_kernel_cpu(args...);},
        idx
    );
}

void uniform_kernel_cpu(auto *tptr, Tensor *t, int i, double a, double b) {
    tptr[i] = a + (b - a) * ((double) rand() / ((double) RAND_MAX + 1.0));
}

void launch_uniform_kernel_cpu(auto *tptr, Tensor *t, int numel, double a, double b) {
    #pragma omp parallel for
    for (int i=0; i < numel; i++)
        uniform_kernel_cpu(tptr, t, i, a, b);
}

void uniform_cpu(Tensor *t, double a, double b){
    dispatch(t,
        [](auto... args){return dispatch_unary_inplace_op(args...);}, 
        [](auto... args){launch_uniform_kernel_cpu(args...);},
        a, b
    );
}


void maximum_kernel_cpu(auto *tptr, auto *t2ptr, auto *resptr, Tensor *t, Tensor *t2, int i) {
    auto v1 = tptr[storage_idx(t, i)];
    auto v2 = t2ptr[storage_idx(t2, i)];
    resptr[i] = v1 > v2 ? v1 : v2;
}

void launch_maximum_kernel_cpu(auto *tptr, auto *t2ptr, auto *resptr, Tensor *t, Tensor *t2, int numel) {
    #pragma omp parallel for
    for (int i=0; i < numel; i++)
        maximum_kernel_cpu(tptr, t2ptr, resptr, t, t2, i);
}

Tensor *maximum_cpu(Tensor *t, Tensor *t2) {
    return dispatch(t, t2,
        [](auto... args){return dispatch_binary_op(args...);}, 
        [](auto... args){launch_maximum_kernel_cpu(args...);}
    );
}


void mul_reduce_kernel_cpu(auto *tptr, auto *t2ptr, auto *resptr, int axis, Tensor *t, Tensor *t2, Tensor *res, int i){
    int t_idx;
    int t2_idx;
    int idx;
    int axis_offset;
    resptr[i] = 0;
    for(int j = 0; j < t->shape[axis]; j++){
        t_idx = 0;
        t2_idx = 0;
        idx = i;
        for(int k = 0; k < t->ndim; k++){
            axis_offset = (k == axis) ? j : 0;
            t_idx += (idx / res->strides[k] + axis_offset) * t->strides[k];
            t2_idx += (idx / res->strides[k] + axis_offset) * t2->strides[k];
            idx %= res->strides[k];
        }
        t_idx += t->offset;
        t2_idx += t2->offset;
        resptr[i] += tptr[t_idx] * t2ptr[t2_idx];
    }
}

void launch_mul_reduce_kernel_cpu(auto *tptr, auto *t2ptr, auto *resptr, int axis, Tensor *t, Tensor *t2, Tensor *res) {
    #pragma omp parallel for
    for (int i=0; i < res->numel; i++)
        mul_reduce_kernel_cpu(tptr, t2ptr, resptr, axis, t, t2, res, i);
}

Tensor *mul_reduce_cpu(Tensor *t, Tensor *t2, int axis) {
    return dispatch(t, t2,
        [](auto... args){return dispatch_binary_reduce_op(args...);}, 
        [](auto... args){launch_mul_reduce_kernel_cpu(args...);},
        axis
    );
}


void pow_kernel_cpu(auto *tptr, double *resptr, Tensor *t, int i, double x){
    resptr[i] = pow(tptr[storage_idx(t, i)], x);
}

void launch_pow_kernel_cpu(auto *tptr, double *resptr, Tensor *t, int numel, double x){
    #pragma omp parallel for
    for (int i=0; i < numel; i++)
        pow_kernel_cpu(tptr, resptr, t, i, x);
}

Tensor *pow_cpu(Tensor *t, double x) {
    return dispatch(t,
        [](auto... args){return dispatch_unary_op_d(args...);},
        [](auto... args){launch_pow_kernel_cpu(args...);},
        x
    );
}


void exp_kernel_cpu(auto *tptr, double *resptr, Tensor *t, int i){
    resptr[i] = exp(tptr[storage_idx(t, i)]);
}

void launch_exp_kernel_cpu(auto *tptr, double *resptr, Tensor *t, int numel){
    #pragma omp parallel for
    for (int i=0; i < numel; i++)
        exp_kernel_cpu(tptr, resptr, t, i);
}

Tensor *exp_cpu(Tensor *t) {
    return dispatch(t,
        [](auto... args){return dispatch_unary_op_d(args...);}, 
        [](auto... args){launch_exp_kernel_cpu(args...);}
    );
}


void log_kernel_cpu(auto *tptr, double *resptr, Tensor *t, int i){
    resptr[i] = log(tptr[storage_idx(t, i)]);
}

void launch_log_kernel_cpu(auto *tptr, double *resptr, Tensor *t, int numel){
    #pragma omp parallel for
    for (int i=0; i < numel; i++)
        log_kernel_cpu(tptr, resptr, t, i);
}

Tensor *log_cpu(Tensor *t) {
    return dispatch(t,
        [](auto... args){return dispatch_unary_op_d(args...);}, 
        [](auto... args){launch_log_kernel_cpu(args...);}
    );
}


void tanh_kernel_cpu(auto *tptr, double *resptr, Tensor *t, int i){
    resptr[i] = tanh(tptr[storage_idx(t, i)]);
}

void launch_tanh_kernel_cpu(auto *tptr, double *resptr, Tensor *t, int numel){
    #pragma omp parallel for
    for (int i=0; i < numel; i++)
        tanh_kernel_cpu(tptr, resptr, t, i);
}

Tensor *tanh_cpu(Tensor *t) {
    return dispatch(t,
        [](auto... args){return dispatch_unary_op_d(args...);}, 
        [](auto... args){launch_tanh_kernel_cpu(args...);}
    );
}


void contiguous_kernel_cpu(auto *tptr, auto *resptr, Tensor *t, int i){
    resptr[i] = tptr[storage_idx(t, i)];
}

void launch_contiguous_kernel_cpu(auto *tptr, auto *resptr, Tensor *t, int numel){
    #pragma omp parallel for
    for (int i=0; i < numel; i++)
        contiguous_kernel_cpu(tptr, resptr, t, i);
}

Tensor *contiguous_cpu(Tensor *t) {
    return dispatch(t,
        [](auto... args){return dispatch_unary_op(args...);}, 
        [](auto... args){launch_contiguous_kernel_cpu(args...);}
    );
}


void arange_kernel_cpu(auto *tptr, Tensor *t, int i, int start, int stop, int step){
    tptr[i] = start + i * step;
}

void launch_arange_kernel_cpu(auto *tptr, Tensor *t, int numel, int start, int stop, int step){
    #pragma omp parallel for
    for (int i=0; i < numel; i++)
        arange_kernel_cpu(tptr, t, i, start, stop, step);
}

Tensor *arange_cpu(Tensor *t, int start, int stop, int step) {
    return dispatch(t,
        [](auto... args){return dispatch_unary_inplace_op(args...);}, 
        [](auto... args){launch_arange_kernel_cpu(args...);},
        start, stop, step
    );
}
