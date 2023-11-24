//
// Copyright (C) 2002-2023 CERN for the benefit of the ATLAS collaboration
//
// Dear emacs, this is -*- c++ -*-
//

#ifndef CALORECGPU_DATAHOLDERS_H
#define CALORECGPU_DATAHOLDERS_H

#include "Helpers.h"
#include "CUDAFriendlyClasses.h"

#define CALORECGPU_USE_PINNED_MEMORY 1

namespace CaloRecGPU
{

  class ConstantDataHolder
  {
   public:

    void sendToGPU(const bool clear_CPU = true);

    CaloRecGPU::Helpers::CPU_object<CaloRecGPU::GeometryArr> m_geometry;

    CaloRecGPU::Helpers::CPU_object<CaloRecGPU::CellNoiseArr> m_cell_noise;

    CaloRecGPU::Helpers::CUDA_object<CaloRecGPU::GeometryArr> m_geometry_dev;

    CaloRecGPU::Helpers::CUDA_object<CaloRecGPU::CellNoiseArr> m_cell_noise_dev;

  };

  class EventDataHolder
  {
   public:

    void sendToGPU(const bool clear_CPU = false,
                   const bool has_state = false,
                   const bool has_clusters = false,
                   const bool has_moments = false);

    void returnToCPU(const bool clear_GPU = false,
                     const bool return_cells = true,
                     const bool return_clusters = true,
                     const bool return_moments = true);
    
    ///This function is asynchronous.
    void returnCellsToCPU(CaloRecGPU::CUDA_Helpers::CUDAStreamPtrHolder stream = {});
    ///This function is asynchronous.
    void returnClustersToCPU(CaloRecGPU::CUDA_Helpers::CUDAStreamPtrHolder stream = {});
    ///This function is asynchronous.
    void returnMomentsToCPU(CaloRecGPU::CUDA_Helpers::CUDAStreamPtrHolder stream = {});

    ///This function is asynchronous.
    void returnClusterNumberToCPU(CaloRecGPU::CUDA_Helpers::CUDAStreamPtrHolder stream = {});

    ///We assume the cluster number is already known
    ///and thus only return `num_clusters` clusters.
    ///This function is asynchronous.
    void returnSomeClustersToCPU(const size_t num_clusters, CaloRecGPU::CUDA_Helpers::CUDAStreamPtrHolder stream = {});

    ///We assume the cluster number is already known
    ///and thus only return `num_clusters` clusters.
    ///This function is asynchronous.
    void returnSomeMomentsToCPU(const size_t num_clusters, CaloRecGPU::CUDA_Helpers::CUDAStreamPtrHolder stream = {});

    void allocate(const bool also_GPU = true);
    
    void clear_GPU();

#if CALORECGPU_USE_PINNED_MEMORY

    CaloRecGPU::Helpers::CUDA_pinned_CPU_object<CaloRecGPU::CellInfoArr> m_cell_info;
    CaloRecGPU::Helpers::CUDA_pinned_CPU_object<CaloRecGPU::CellStateArr> m_cell_state;
    CaloRecGPU::Helpers::CUDA_pinned_CPU_object<CaloRecGPU::ClusterInfoArr> m_clusters;
    CaloRecGPU::Helpers::CUDA_pinned_CPU_object<CaloRecGPU::ClusterMomentsArr> m_moments;

#else

    CaloRecGPU::Helpers::CPU_object<CaloRecGPU::CellInfoArr> m_cell_info;
    CaloRecGPU::Helpers::CPU_object<CaloRecGPU::CellStateArr> m_cell_state;
    CaloRecGPU::Helpers::CPU_object<CaloRecGPU::ClusterInfoArr> m_clusters;
    CaloRecGPU::Helpers::CPU_object<CaloRecGPU::ClusterMomentsArr> m_moments;

#endif

    CaloRecGPU::Helpers::CUDA_object<CaloRecGPU::CellInfoArr> m_cell_info_dev;
    CaloRecGPU::Helpers::CUDA_object<CaloRecGPU::CellStateArr> m_cell_state_dev;
    CaloRecGPU::Helpers::CUDA_object<CaloRecGPU::ClusterInfoArr> m_clusters_dev;
    CaloRecGPU::Helpers::CUDA_object<CaloRecGPU::ClusterMomentsArr> m_moments_dev;

  };

}

#endif //CALORECGPU_DATAHOLDERS_H