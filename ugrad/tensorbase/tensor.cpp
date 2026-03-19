#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tensor.h"
#include "cpu.h"

#ifndef CPU_ONLY
#include "cuda.h"
#endif


extern "C"
Storage *create_storage(int nbytes, int device){
    if (device == CPU_DEVICE)
        return create_cpu_storage(nbytes);
    #ifdef CUDA_H
    else
        return create_cuda_storage(nbytes, device);
    #endif
    return NULL;
}

Storage *clone_storage(Storage *s){
    if (s->device == CPU_DEVICE)
        return clone_cpu_storage(s);
    #ifdef CUDA_H
    else
        return clone_cuda_storage(s);
    #endif
    return NULL;
}

extern "C"
Storage *cc_storage(int nbytes, void *data){
    Storage *storage = create_storage(nbytes, CPU_DEVICE);
    memcpy(storage->data, data, storage->nbytes);
    return storage;
}

extern "C"
Tensor *create_tensor(Storage *storage, int *shape, int ndim, int dtype){
    Tensor *t = (Tensor *) malloc(sizeof(Tensor));
    if (t == NULL) {
        perror("malloc");
        exit(1);
    }

    t->offset = 0;
    t->ndim = ndim;
    t->dtype = dtype;
    t->numel = 1;
    t->device = storage->device;

    t->shape = (int *) malloc(t->ndim * sizeof(int));
    if (t->shape == NULL && t->ndim) {
        perror("malloc");
        exit(1);
    }

    t->strides = (int *) malloc(t->ndim * sizeof(int));
    if (t->strides == NULL && t->ndim) {
        perror("malloc");
        exit(1);
    }

    int stride = 1;    
    for (int i = t->ndim - 1; i >= 0; i--) {
        t->shape[i] = shape[i];
        t->numel *= shape[i];
        t->strides[i] = stride;
        stride *= shape[i];
    }

    t->storage = storage;
    t->storage->nshares += 1;

    return t;
}

extern "C"
void delete_tensor(Tensor *t){
    free(t->shape);
    free(t->strides);
    if(t->storage->nshares == 0){
        if(t->storage->device == CPU_DEVICE)
            free(t->storage->data);
        #ifdef CUDA_H
        else
            cuda_free(t->storage->data);
        #endif
        free(t->storage);
    }
    else t->storage->nshares -= 1;
    free(t);
}

extern "C"
void update_tensor(Tensor *t, int *shape, int *strides, int ndim){
    free(t->shape);
    t->ndim = ndim;
    t->shape = (int *) malloc(t->ndim * sizeof(int));
    if (t->shape == NULL && t->ndim) {
        perror("malloc");
        exit(1);
    }
    t->numel = 1;
    for (int i = t->ndim - 1; i >= 0; i--) {
        t->shape[i] = shape[i];
        t->numel *= shape[i];
    }

    free(t->strides);
    t->strides = (int *) malloc(t->ndim * sizeof(int));
    if (t->strides == NULL && t->ndim) {
        perror("malloc");
        exit(1);
    }
    for (int i = t->ndim - 1; i >= 0; i--)
        t->strides[i] = strides[i];
}

extern "C"
Tensor *clone_tensor(Tensor *t){
    Tensor *res = (Tensor *) malloc(sizeof(Tensor));
    if (res == NULL) {
        perror("malloc");
        exit(1);
    }

    res->offset = t->offset;
    res->ndim = t->ndim;
    res->dtype = t->dtype;
    res->numel = t->numel;
    res->device = t->device;

    res->shape = (int *) malloc(res->ndim * sizeof(int));
    if (res->shape == NULL && res->ndim) {
        perror("malloc");
        exit(1);
    }

    memcpy(res->shape, t->shape, res->ndim * sizeof(int));

    res->strides = (int *) malloc(res->ndim * sizeof(int));
    if (res->strides == NULL && res->ndim) {
        perror("malloc");
        exit(1);
    }

    memcpy(res->strides, t->strides, res->ndim * sizeof(int));

    res->storage = clone_storage(t->storage);
    res->storage->nshares += 1;

    return res;
}

extern "C"
Tensor *to(Tensor *t, int device){
    #ifdef CUDA_H
    return (device == CPU_DEVICE) ? to_cpu(t) : to_cuda(t, device);
    #endif
    return t;        
}

extern "C"
void *get_item(Tensor *t, int idx){
    if(idx < t->numel){
        if(t->device == CPU_DEVICE)
            return get_item_cpu(t, storage_idx(t, idx));
        #ifdef CUDA_H
        else
            return get_item_cuda(t, storage_idx(t, idx));
        #endif
    }
    return NULL;
}

extern "C"
bool is_contiguous(Tensor *t){
    int stride = 1;    
    for (int i = t->ndim - 1; i >= 0; i--) {
        if(t->strides[i] != stride)
            return false;
        stride *= t->shape[i];
    }
    return true;
}

Tensor *sum_along_axis(Tensor *t, int axis){
    if(t->device == CPU_DEVICE)
        return sum_cpu(t, axis);
    #ifdef CUDA_H
    else
        return sum_cuda(t, axis);
    #endif

    return NULL;
}

extern "C"
Tensor *sum(Tensor *t, int *axis, int axis_size){
    Tensor *res = create_tensor(t->storage, t->shape,  t->ndim, t->dtype);
    Tensor *prev_res;

    for(int i = 0; i < axis_size; i++){
        prev_res = res;
        res = sum_along_axis(res, axis[i]);
        delete_tensor(prev_res);
    }
    
    return res;
}

Tensor *max_along_axis(Tensor *t, int axis){
    if(t->device == CPU_DEVICE)
        return max_cpu(t, axis);
    #ifdef CUDA_H
    else
        return max_cuda(t, axis);
    #endif

    return NULL;
}

extern "C"
Tensor *max_t(Tensor *t, int *axis, int axis_size){
    Tensor *res = create_tensor(t->storage, t->shape,  t->ndim, t->dtype);
    Tensor *prev_res;

    for(int i = 0; i < axis_size; i++){
        prev_res = res;
        res = max_along_axis(res, axis[i]);
        delete_tensor(prev_res);
    }
    
    return res;
}

extern "C"
Tensor *argmax(Tensor *t, int axis){
    if(t->device == CPU_DEVICE)
        return argmax_cpu(t, axis);
    #ifdef CUDA_H
    else
        return argmax_cuda(t, axis);
    #endif

    return NULL;
}

extern "C"
Tensor *gt(Tensor *t, Tensor *t2){
    if(t->device == CPU_DEVICE)
        return gt_cpu(t, t2);
    #ifdef CUDA_H
    else
        return gt_cuda(t, t2);
    #endif

    return NULL;
}

extern "C"
Tensor *eq(Tensor *t, Tensor *t2){
    if(t->device == CPU_DEVICE)
        return eq_cpu(t, t2);
    #ifdef CUDA_H
    else
        return eq_cuda(t, t2);
    #endif

    return NULL;
}

extern "C"
Tensor *add(Tensor *t, Tensor *t2){
    if(t->device == CPU_DEVICE)
        return add_cpu(t, t2);
    #ifdef CUDA_H
    else
        return add_cuda(t, t2);
    #endif

    return NULL;
}

extern "C"
Tensor *mul(Tensor *t, Tensor *t2){
    if(t->device == CPU_DEVICE)
        return mul_cpu(t, t2);
    #ifdef CUDA_H
    else
        return mul_cuda(t, t2);
    #endif

    return NULL;
}

extern "C"
void assign(Tensor *t, Tensor *t2){
    if(t->device == CPU_DEVICE)
        assign_cpu(t, t2);
    #ifdef CUDA_H
    else
        assign_cuda(t, t2);
    #endif
}

extern "C"
void add_at(Tensor *t, Tensor *idx, Tensor *t2){
    if(t->device == CPU_DEVICE)
        add_at_cpu(t, idx, t2);
    #ifdef CUDA_H
    else
        add_at_cuda(t, idx, t2);
    #endif

}

extern "C"
void uniform(Tensor *t, double a, double b){
    if(t->device == CPU_DEVICE)
        uniform_cpu(t, a, b);
    #ifdef CUDA_H
    else
        uniform_cuda(t, a, b);
    #endif

}

extern "C"
Tensor *maximum(Tensor *t, Tensor *t2){
    if(t->device == CPU_DEVICE)
        return maximum_cpu(t, t2);
    #ifdef CUDA_H
    else
        return maximum_cuda(t, t2);
    #endif

    return NULL;
}

extern "C"
Tensor *mul_reduce(Tensor *t, Tensor *t2, int axis){
    if(t->device == CPU_DEVICE)
        return mul_reduce_cpu(t, t2, axis);
    #ifdef CUDA_H
    else
        return mul_reduce_cuda(t, t2, axis);
    #endif

    return NULL;
}

extern "C"
Tensor *pow_t(Tensor *t, double x){
    if(t->device == CPU_DEVICE)
        return pow_cpu(t, x);
    #ifdef CUDA_H
    else
        return pow_cuda(t, x);
    #endif

    return NULL;
}

extern "C"
Tensor *exp_t(Tensor *t){
    if(t->device == CPU_DEVICE)
        return exp_cpu(t);
    #ifdef CUDA_H
    else
        return exp_cuda(t);
    #endif

    return NULL;
}

extern "C"
Tensor *log_t(Tensor *t){
    if(t->device == CPU_DEVICE)
        return log_cpu(t);
    #ifdef CUDA_H
    else
        return log_cuda(t);
    #endif

    return NULL;
}

extern "C"
Tensor *tanh_t(Tensor *t){
    if(t->device == CPU_DEVICE)
        return tanh_cpu(t);
    #ifdef CUDA_H
    else
        return tanh_cuda(t);
    #endif

    return NULL;
}

extern "C"
Tensor *contiguous(Tensor *t){
    if(t->device == CPU_DEVICE)
        return contiguous_cpu(t);
    #ifdef CUDA_H
    else
        return contiguous_cuda(t);
    #endif

    return NULL;
}

extern "C"
Tensor *arange(int start, int stop, int step, int device){
    int numel = (stop - start) / step + (((stop - start) % step) ? 1 : 0);
    int shape[] = {numel};

    Storage *storage = create_storage(sizeof(long) * numel, device);
    Tensor *t = create_tensor(storage, shape, 1, LONG_DTYPE);

    if(device == CPU_DEVICE)
        return arange_cpu(t, start, stop, step);
    #ifdef CUDA_H
    else
        return arange_cuda(t, start, stop, step);
    #endif

    return NULL;
}