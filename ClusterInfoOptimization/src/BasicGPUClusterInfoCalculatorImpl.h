//
// Copyright (C) 2002-2023 CERN for the benefit of the ATLAS collaboration
//
// Dear emacs, this is -*- c++ -*-
//

#ifndef CALORECGPU_BASICGPUCLUSTERINFOCALCULATORIMPL_H
#define CALORECGPU_BASICGPUCLUSTERINFOCALCULATORIMPL_H

#include "CUDAFriendlyClasses.h"
#include "DataHolders.h"

#include "IGPUKernelSizeOptimizer.h"

namespace BasicClusterInfoCalculator
{

  struct ClusterInfoCalculatorTemporaries
  {
    float seedCellPhi[CaloRecGPU::NMaxClusters];
  };

  void register_kernels(IGPUKernelSizeOptimizer & optimizer);

  void updateSeedCellProperties(CaloRecGPU::EventDataHolder & holder,
                                CaloRecGPU::Helpers::CUDA_kernel_object<ClusterInfoCalculatorTemporaries> temps,
                                const CaloRecGPU::ConstantDataHolder & instance_data,
                                const IGPUKernelSizeOptimizer & optimizer,
                                const bool synchronize = false,
                                CaloRecGPU::CUDA_Helpers::CUDAStreamPtrHolder stream_to_use = {});

  void calculateClusterProperties(CaloRecGPU::EventDataHolder & holder, CaloRecGPU::Helpers::CUDA_kernel_object<ClusterInfoCalculatorTemporaries> temps,
                                  const CaloRecGPU::ConstantDataHolder & instance_data,
                                  const IGPUKernelSizeOptimizer & optimizer,
                                  const bool synchronize = false,
                                  const bool cut_in_absolute_ET = true, const float absolute_ET_threshold = -1,
                                  CaloRecGPU::CUDA_Helpers::CUDAStreamPtrHolder stream_to_use = {});

}
#endif //CALORECGPU_BASICGPUCLUSTERINFOCALCULATORIMPL_H