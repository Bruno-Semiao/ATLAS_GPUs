//
// Copyright (C) 2002-2023 CERN for the benefit of the ATLAS collaboration
//
// Dear emacs, this is -*- c++ -*-
//

#include "Helpers.h"

void * CaloRecGPU::CUDA_Helpers::allocate(const size_t num)
{
  void * ret;
  CUDA_ERRCHECK(cudaMalloc(&ret, num));
  return ret;
}

void CaloRecGPU::CUDA_Helpers::deallocate(void * address)
{
  CUDA_ERRCHECK(cudaFree(address));
}


void * CaloRecGPU::CUDA_Helpers::allocate_pinned(const size_t num)
{
  void * ret;
  CUDA_ERRCHECK(cudaMallocHost(&ret, num));
  return ret;
}

void CaloRecGPU::CUDA_Helpers::deallocate_pinned(void * address)
{
  CUDA_ERRCHECK(cudaFreeHost(address));
}


void CaloRecGPU::CUDA_Helpers::GPU_to_CPU(void * dest, const void * const source, const size_t num)
{
  CUDA_ERRCHECK(cudaMemcpy(dest, source, num, cudaMemcpyDeviceToHost));
}

void CaloRecGPU::CUDA_Helpers::CPU_to_GPU(void * dest, const void * const source, const size_t num)
{
  CUDA_ERRCHECK(cudaMemcpy(dest, source, num, cudaMemcpyHostToDevice));
}

void CaloRecGPU::CUDA_Helpers::GPU_to_GPU(void * dest, const void * const source, const size_t num)
{
  CUDA_ERRCHECK(cudaMemcpy(dest, source, num, cudaMemcpyDeviceToDevice));
}



void CaloRecGPU::CUDA_Helpers::GPU_to_CPU_async(void * dest, const void * const source, const size_t num, CaloRecGPU::CUDA_Helpers::CUDAStreamPtrHolder stream)
{
  const cudaStream_t & stream_to_use = (stream != nullptr ? * ((cudaStream_t *) stream) : cudaStreamPerThread);
  CUDA_ERRCHECK(cudaMemcpyAsync(dest, source, num, cudaMemcpyDeviceToHost, cudaStreamPerThread));
}

void CaloRecGPU::CUDA_Helpers::CPU_to_GPU_async(void * dest, const void * const source, const size_t num, CaloRecGPU::CUDA_Helpers::CUDAStreamPtrHolder stream)
{
  const cudaStream_t & stream_to_use = (stream != nullptr ? * ((cudaStream_t *) stream) : cudaStreamPerThread);
  CUDA_ERRCHECK(cudaMemcpyAsync(dest, source, num, cudaMemcpyHostToDevice, stream_to_use));
}

void CaloRecGPU::CUDA_Helpers::GPU_to_GPU_async(void * dest, const void * const source, const size_t num, CaloRecGPU::CUDA_Helpers::CUDAStreamPtrHolder stream)
{
  const cudaStream_t & stream_to_use = (stream != nullptr ? * ((cudaStream_t *) stream) : cudaStreamPerThread);
  CUDA_ERRCHECK(cudaMemcpyAsync(dest, source, num, cudaMemcpyDeviceToDevice, stream_to_use));
}

void CaloRecGPU::CUDA_Helpers::GPU_synchronize(CaloRecGPU::CUDA_Helpers::CUDAStreamPtrHolder stream)
{
  const cudaStream_t & stream_to_use = (stream != nullptr ? * ((cudaStream_t *) stream) : cudaStreamPerThread);
  CUDA_ERRCHECK(cudaStreamSynchronize(stream_to_use));
}
