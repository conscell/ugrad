#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "dispatch.h"


void inline checkCudaError() {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}

Tensor *to_cuda(Tensor *t, int device) {
    Storage *storage = create_storage(t->storage->nbytes, device);
    Tensor *res = create_tensor(storage, t->shape, t->ndim, t->dtype);

    cudaMemcpy(res->storage->data, t->storage->data, res->storage->nbytes, cudaMemcpyHostToDevice);
    checkCudaError();

    return res;
}

Tensor *to_cpu(Tensor *t) {
    Storage *storage = create_storage(t->storage->nbytes, CPU_DEVICE);
    Tensor *res = create_tensor(storage, t->shape, t->ndim, t->dtype);

    cudaMemcpy(res->storage->data, t->storage->data, res->storage->nbytes, cudaMemcpyDeviceToHost);
    checkCudaError();

    return res;
}

void cuda_free(void *ptr){
    cudaFree(ptr);
    checkCudaError();
}

Storage *create_cuda_storage(int nbytes, int device){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (device >= deviceCount) {
        fprintf(stderr, "Failed to create storage on device %d; only %d devices are available\n", device, deviceCount);
        exit(1);
    }
    cudaSetDevice(device); 

    Storage *storage = (Storage *) malloc(sizeof(Storage));
    if (storage == NULL) {
        perror("malloc");
        exit(1);
    }
    storage->nbytes = nbytes;
    storage->nshares = -1;
    storage->device = device;
    
    cudaMalloc((void **)&storage->data, storage->nbytes);
    checkCudaError();

    return storage;
}

Storage *clone_cuda_storage(Storage *s){
    Storage *storage = create_cuda_storage(s->nbytes, s->device);
    cudaMemcpy(storage->data, s->data, storage->nbytes, cudaMemcpyDeviceToDevice);
    checkCudaError();
    
    return storage;
}

__host__ TensorParams *copy_tensor_params(Tensor *t){
    TensorParams *tp = (TensorParams *) malloc(sizeof(TensorParams));
    if (tp == NULL) {
        perror("malloc");
        exit(1);
    }

    tp->offset = t->offset;
    tp->ndim = t->ndim;
    tp->numel = t->numel;

    cudaMalloc((void **)&tp->shape, tp->ndim * sizeof(int));
    cudaMemcpy(tp->shape, t->shape, tp->ndim * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&tp->strides, tp->ndim * sizeof(int));
    cudaMemcpy(tp->strides, t->strides, tp->ndim * sizeof(int), cudaMemcpyHostToDevice);

    TensorParams *tpcuda;

    cudaMalloc((void **)&tpcuda, sizeof(TensorParams));
    cudaMemcpy(tpcuda, tp, sizeof(TensorParams), cudaMemcpyHostToDevice);
    free(tp);
    checkCudaError();

    return tpcuda;

}

__device__ int cu_storage_idx(TensorParams *t, int logical_idx){
    int idx = 0;
    int logical_stride = t->numel;

    for (int i = 0; i < t->ndim; i++){
        logical_stride /= t->shape[i];
        idx += (logical_idx / logical_stride) * t->strides[i];
        logical_idx %= logical_stride;
    }
    return idx + t->offset;
}

void *get_item_cuda(Tensor *t, int idx){
    void *data = NULL;
    switch(t->dtype) {
        case DOUBLE_DTYPE: {
            data = malloc(sizeof(double));
            if (data == NULL) {
                perror("malloc");
                exit(1);
            }
            cudaMemcpy(data, (void *) ((double *) t->storage->data + idx), sizeof(double), cudaMemcpyDeviceToHost);
            break;
        }
        case LONG_DTYPE: {
            data = malloc(sizeof(long));
            if (data == NULL) {
                perror("malloc");
                exit(1);
            }
            cudaMemcpy(data, (void *) ((long *) t->storage->data + idx), sizeof(long), cudaMemcpyDeviceToHost);
            break;
        }

    }
    return data;
}


__global__ void sum_kernel_cuda(auto *tptr, auto *resptr, int axis, TensorParams *t, TensorParams *res){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < res->numel) {
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
}

void launch_sum_kernel_cuda(auto *tptr, auto *resptr, int axis, Tensor *t, Tensor *res) {
    TensorParams *tp = copy_tensor_params(t), *resp = copy_tensor_params(res);
    int number_of_blocks = (res->numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    sum_kernel_cuda<<<number_of_blocks, THREADS_PER_BLOCK>>>(tptr, resptr, axis, tp, resp);

    cudaDeviceSynchronize();
    cudaFree(tp);
    cudaFree(resp);
    checkCudaError();
}

Tensor *sum_cuda(Tensor *t, int axis){
    return dispatch(t,
        [](auto... args){return dispatch_unary_reduce_op(args...);},
        [](auto... args){launch_sum_kernel_cuda(args...);},
        axis
    );
}


__global__ void max_kernel_cuda(double *tptr, double *resptr, int axis, TensorParams *t, TensorParams *res){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < res->numel) {
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
}

__global__ void max_kernel_cuda(long *tptr, long *resptr, int axis, TensorParams *t, TensorParams *res){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < res->numel) {
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
}

void launch_max_kernel_cuda(auto *tptr, auto *resptr, int axis, Tensor *t, Tensor *res) {
    TensorParams *tp = copy_tensor_params(t), *resp = copy_tensor_params(res);
    int number_of_blocks = (res->numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    max_kernel_cuda<<<number_of_blocks, THREADS_PER_BLOCK>>>(tptr, resptr, axis, tp, resp);

    cudaDeviceSynchronize();
    cudaFree(tp);
    cudaFree(resp);
    checkCudaError();
}

Tensor *max_cuda(Tensor *t, int axis){
    return dispatch(t,
        [](auto... args){return dispatch_unary_reduce_op(args...);},
        [](auto... args){launch_max_kernel_cuda(args...);},
        axis
    );
}


__global__ void argmax_kernel_cuda(double *tptr, long *resptr, double *max_tptr, int axis, TensorParams *t, TensorParams *res){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < res->numel) {
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
}

__global__ void argmax_kernel_cuda(long *tptr, long *resptr, long *max_tptr, int axis, TensorParams *t, TensorParams *res){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < res->numel) {
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
}

void launch_argmax_kernel_cuda(auto *tptr, long *resptr, auto *max_tptr, int axis, Tensor *t, Tensor *res){
    TensorParams *tp = copy_tensor_params(t), *resp = copy_tensor_params(res);
    int number_of_blocks = (res->numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    argmax_kernel_cuda<<<number_of_blocks, THREADS_PER_BLOCK>>>(tptr, resptr, max_tptr, axis, tp, resp);

    cudaDeviceSynchronize();
    cudaFree(tp);
    cudaFree(resp);
    checkCudaError();
}

Tensor *argmax_cuda(Tensor *t, int axis){
    return dispatch(t,
        [](auto... args){return dispatch_argmax(args...);},
        [](auto... args){launch_argmax_kernel_cuda(args...);},
        axis
    );
}


__global__ void gt_kernel_cuda(auto *tptr, auto *t2ptr, long *resptr, TensorParams *t, TensorParams *t2, int numel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        resptr[i] = tptr[cu_storage_idx(t, i)] > t2ptr[cu_storage_idx(t2, i)] ? 1 : 0;
    }
}

void launch_gt_kernel_cuda(auto *tptr, auto *t2ptr, long *resptr, Tensor *t, Tensor *t2, int numel) {
    TensorParams *tp = copy_tensor_params(t), *tp2 = copy_tensor_params(t2);
    int number_of_blocks = (numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    gt_kernel_cuda<<<number_of_blocks, THREADS_PER_BLOCK>>>(tptr, t2ptr, resptr, tp, tp2, numel);
    
    cudaDeviceSynchronize();
    cudaFree(tp);
    cudaFree(tp2);
    checkCudaError();
}

Tensor *gt_cuda(Tensor *t, Tensor *t2) {
    return dispatch(t, t2,
        [](auto... args){return dispatch_binary_op_l(args...);},
        [](auto... args){launch_gt_kernel_cuda(args...);}
    );
}


__global__ void eq_kernel_cuda(auto *tptr, auto *t2ptr, long *resptr, TensorParams *t, TensorParams *t2, int numel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        resptr[i] = tptr[cu_storage_idx(t, i)] == t2ptr[cu_storage_idx(t2, i)] ? 1 : 0;
    }
}

void launch_eq_kernel_cuda(auto *tptr, auto *t2ptr, long *resptr, Tensor *t, Tensor *t2, int numel) {
    TensorParams *tp = copy_tensor_params(t), *tp2 = copy_tensor_params(t2);
    int number_of_blocks = (numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    eq_kernel_cuda<<<number_of_blocks, THREADS_PER_BLOCK>>>(tptr, t2ptr, resptr, tp, tp2, numel);
    
    cudaDeviceSynchronize();
    cudaFree(tp);
    cudaFree(tp2);
    checkCudaError();
}

Tensor *eq_cuda(Tensor *t, Tensor *t2) {
    return dispatch(t, t2,
        [](auto... args){return dispatch_binary_op_l(args...);},
        [](auto... args){launch_eq_kernel_cuda(args...);}
    );
}


__global__ void add_kernel_cuda(auto *tptr, auto *t2ptr, auto *resptr, TensorParams *t, TensorParams *t2, int numel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        resptr[i] = tptr[cu_storage_idx(t, i)] + t2ptr[cu_storage_idx(t2, i)];
    }
}

void launch_add_kernel_cuda(auto *tptr, auto *t2ptr, auto *resptr, Tensor *t, Tensor *t2, int numel) {
    TensorParams *tp = copy_tensor_params(t), *tp2 = copy_tensor_params(t2);
    int number_of_blocks = (numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    add_kernel_cuda<<<number_of_blocks, THREADS_PER_BLOCK>>>(tptr, t2ptr, resptr, tp, tp2, numel);
    
    cudaDeviceSynchronize();
    cudaFree(tp);
    cudaFree(tp2);
    checkCudaError();
}

Tensor *add_cuda(Tensor *t, Tensor *t2) {
    return dispatch(t, t2,
        [](auto... args){return dispatch_binary_op(args...);},
        [](auto... args){launch_add_kernel_cuda(args...);}
    );
}


__global__ void mul_kernel_cuda(auto *tptr, auto *t2ptr, auto *resptr, TensorParams *t, TensorParams *t2, int numel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        resptr[i] = tptr[cu_storage_idx(t, i)] * t2ptr[cu_storage_idx(t2, i)];
    }
}

void launch_mul_kernel_cuda(auto *tptr, auto *t2ptr, auto *resptr, Tensor *t, Tensor *t2, int numel) {
    TensorParams *tp = copy_tensor_params(t), *tp2 = copy_tensor_params(t2);
    int number_of_blocks = (numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    mul_kernel_cuda<<<number_of_blocks, THREADS_PER_BLOCK>>>(tptr, t2ptr, resptr, tp, tp2, numel);
    
    cudaDeviceSynchronize();
    cudaFree(tp);
    cudaFree(tp2);
    checkCudaError();
}

Tensor *mul_cuda(Tensor *t, Tensor *t2) {
    return dispatch(t, t2,
        [](auto... args){return dispatch_binary_op(args...);},
        [](auto... args){launch_mul_kernel_cuda(args...);}
    );
}


__global__ void maximum_kernel_cuda(auto *tptr, auto *t2ptr, auto *resptr, TensorParams *t, TensorParams *t2, int numel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        auto v1 = tptr[cu_storage_idx(t, i)];
        auto v2 = t2ptr[cu_storage_idx(t2, i)];
        resptr[i] = v1 > v2 ? v1 : v2;
    }
}

void launch_maximum_kernel_cuda(auto *tptr, auto *t2ptr, auto *resptr, Tensor *t, Tensor *t2, int numel) {
    TensorParams *tp = copy_tensor_params(t), *tp2 = copy_tensor_params(t2);
    int number_of_blocks = (numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    maximum_kernel_cuda<<<number_of_blocks, THREADS_PER_BLOCK>>>(tptr, t2ptr, resptr, tp, tp2, numel);
    
    cudaDeviceSynchronize();
    cudaFree(tp);
    cudaFree(tp2);
    checkCudaError();
}

Tensor *maximum_cuda(Tensor *t, Tensor *t2) {
    return dispatch(t, t2,
        [](auto... args){return dispatch_binary_op(args...);},
        [](auto... args){launch_maximum_kernel_cuda(args...);}
    );
}


__global__ void mul_reduce_kernel_cuda(auto *tptr, auto *t2ptr, auto *resptr, int axis, TensorParams *t, TensorParams *t2, TensorParams *res){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < res->numel) {
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
}

void launch_mul_reduce_kernel_cuda(auto *tptr, auto *t2ptr, auto *resptr, int axis, Tensor *t, Tensor *t2, Tensor *res) {
    TensorParams *tp = copy_tensor_params(t), *tp2 = copy_tensor_params(t2), *resp = copy_tensor_params(res);
    int number_of_blocks = (res->numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    mul_reduce_kernel_cuda<<<number_of_blocks, THREADS_PER_BLOCK>>>(tptr, t2ptr, resptr, axis, tp, tp2, resp);
    
    cudaDeviceSynchronize();
    cudaFree(tp);
    cudaFree(tp2);
    cudaFree(resp);
    checkCudaError();
}

Tensor *mul_reduce_cuda(Tensor *t, Tensor *t2, int axis) {
    return dispatch(t, t2,
        [](auto... args){return dispatch_binary_reduce_op(args...);},
        [](auto... args){launch_mul_reduce_kernel_cuda(args...);},
        axis
    );
}


__global__ void pow_kernel_cuda(auto *tptr, double *resptr, double x, TensorParams *t, int numel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        resptr[i] = pow((double)tptr[cu_storage_idx(t, i)], x);
    }
}

void launch_pow_kernel_cuda(auto *tptr, double *resptr, double x, Tensor *t, int numel) {
    TensorParams *tp = copy_tensor_params(t);
    int number_of_blocks = (numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    pow_kernel_cuda<<<number_of_blocks, THREADS_PER_BLOCK>>>(tptr, resptr, x, tp, numel);

    cudaDeviceSynchronize();
    cudaFree(tp);
    checkCudaError();
}

Tensor *pow_cuda(Tensor *t, double x) {
    return dispatch(t,
        [](auto... args){return dispatch_unary_op_d_xtra(args...);}, 
        [](auto... args){launch_pow_kernel_cuda(args...);},
        x
    );
}


__global__ void exp_kernel_cuda(auto *tptr, double *resptr, TensorParams *t, int numel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        resptr[i] = exp((double)tptr[cu_storage_idx(t, i)]);
    }
}

void launch_exp_kernel_cuda(auto *tptr, double *resptr, Tensor *t, int numel) {
    TensorParams *tp = copy_tensor_params(t);
    int number_of_blocks = (numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    exp_kernel_cuda<<<number_of_blocks, THREADS_PER_BLOCK>>>(tptr, resptr, tp, numel);

    cudaDeviceSynchronize();
    cudaFree(tp);
    checkCudaError();
}

Tensor *exp_cuda(Tensor *t) {
    return dispatch(t,
        [](auto... args){return dispatch_unary_op_d(args...);}, 
        [](auto... args){launch_exp_kernel_cuda(args...);}
    );
}


__global__ void log_kernel_cuda(auto *tptr, double *resptr, TensorParams *t, int numel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        resptr[i] = log((double)tptr[cu_storage_idx(t, i)]);
    }
}

void launch_log_kernel_cuda(auto *tptr, double *resptr, Tensor *t, int numel) {
    TensorParams *tp = copy_tensor_params(t);
    int number_of_blocks = (numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    log_kernel_cuda<<<number_of_blocks, THREADS_PER_BLOCK>>>(tptr, resptr, tp, numel);

    cudaDeviceSynchronize();
    cudaFree(tp);
    checkCudaError();
}

Tensor *log_cuda(Tensor *t) {
    return dispatch(t,
        [](auto... args){return dispatch_unary_op_d(args...);}, 
        [](auto... args){launch_log_kernel_cuda(args...);}
    );
}


__global__ void tanh_kernel_cuda(auto *tptr, double *resptr, TensorParams *t, int numel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        resptr[i] = tanh((double)tptr[cu_storage_idx(t, i)]);
    }
}

void launch_tanh_kernel_cuda(auto *tptr, double *resptr, Tensor *t, int numel) {
    TensorParams *tp = copy_tensor_params(t);
    int number_of_blocks = (numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    tanh_kernel_cuda<<<number_of_blocks, THREADS_PER_BLOCK>>>(tptr, resptr, tp, numel);

    cudaDeviceSynchronize();
    cudaFree(tp);
    checkCudaError();
}

Tensor *tanh_cuda(Tensor *t) {
    return dispatch(t,
        [](auto... args){return dispatch_unary_op_d(args...);}, 
        [](auto... args){launch_tanh_kernel_cuda(args...);}
    );
}


__global__ void contiguous_kernel_cuda(auto *tptr, auto *resptr, TensorParams *t, int numel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        resptr[i] = tptr[cu_storage_idx(t, i)];
    }
}

void launch_contiguous_kernel_cuda(auto *tptr, auto *resptr, Tensor *t, int numel) {
    TensorParams *tp = copy_tensor_params(t);
    int number_of_blocks = (numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    contiguous_kernel_cuda<<<number_of_blocks, THREADS_PER_BLOCK>>>(tptr, resptr, tp, numel);

    cudaDeviceSynchronize();
    cudaFree(tp);
    checkCudaError();
}

Tensor *contiguous_cuda(Tensor *t) {
    return dispatch(t,
        [](auto... args){return dispatch_unary_op(args...);}, 
        [](auto... args){launch_contiguous_kernel_cuda(args...);}
    );
}
