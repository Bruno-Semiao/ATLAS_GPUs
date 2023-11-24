//
// Copyright (C) 2002-2023 CERN for the benefit of the ATLAS collaboration
//
// Dear emacs, this is -*- c++ -*-
//

#include "DataHolders.h"

#include "MacroHelpers.h"

void CaloRecGPU::ConstantDataHolder::sendToGPU(const bool clear_CPU)
{
  m_cell_noise_dev = m_cell_noise;
  m_geometry_dev = m_geometry;
  if (clear_CPU)
    {
      m_cell_noise.clear();
      m_geometry.clear();
    }
}

void CaloRecGPU::EventDataHolder::sendToGPU(const bool clear_CPU,
                                            const bool has_state,
                                            const bool has_clusters,
                                            const bool has_moments)
{
  m_cell_info_dev = m_cell_info;
  if (has_state)
    {
      m_cell_state_dev = m_cell_state;
    }
  else
    {
      m_cell_state_dev.allocate();
    }
  if (has_clusters)
    {
      m_clusters_dev = m_clusters;
    }
  else
    {
      m_clusters_dev.allocate();
    }
  if (has_moments)
    {
      m_moments_dev = m_moments;
    }
  else
    {
      m_moments_dev.allocate();
    }

  if (!has_clusters)
    {
      cudaMemset(&(m_clusters_dev->number), 0, sizeof(m_clusters_dev->number));
    }
  //We're not doing this through cudaMemsetAsync because it is reasonable to expect
  //the clusters to be fully sent before doing any more operations.

  if (clear_CPU)
    {
      m_cell_info.clear();
      m_cell_state.clear();
      m_moments.clear();
    }
}

void CaloRecGPU::EventDataHolder::returnToCPU(const bool clear_GPU,
                                              const bool return_cells,
                                              const bool return_clusters,
                                              const bool return_moments)
{
  if (return_cells)
    {
      m_cell_state = m_cell_state_dev;
    }
  if (return_clusters)
    {
      m_clusters = m_clusters_dev;
    }
  if (return_moments)
    {
      m_moments = m_moments_dev;
    }
  if (clear_GPU)
    {
      m_cell_state_dev.clear();
      m_clusters_dev.clear();
      m_cell_info_dev.clear();
      m_moments_dev.clear();
    }
}



void CaloRecGPU::EventDataHolder::returnCellsToCPU(CaloRecGPU::CUDA_Helpers::CUDAStreamPtrHolder stream)
{
  const cudaStream_t & stream_to_use = (stream != nullptr ? * ((cudaStream_t *) stream) : cudaStreamPerThread);
  cudaMemcpyAsync((CaloRecGPU::CellStateArr *) m_cell_state,
                  (CaloRecGPU::CellStateArr *) m_cell_state_dev,
                  sizeof(CaloRecGPU::CellStateArr),
                  cudaMemcpyDeviceToHost, stream_to_use);
}

void CaloRecGPU::EventDataHolder::returnClustersToCPU(CaloRecGPU::CUDA_Helpers::CUDAStreamPtrHolder stream)
{
  const cudaStream_t & stream_to_use = (stream != nullptr ? * ((cudaStream_t *) stream) : cudaStreamPerThread);
  cudaMemcpyAsync((CaloRecGPU::ClusterInfoArr *) m_clusters,
                  (CaloRecGPU::ClusterInfoArr *) m_clusters_dev,
                  sizeof(CaloRecGPU::ClusterInfoArr),
                  cudaMemcpyDeviceToHost, stream_to_use);
}

void CaloRecGPU::EventDataHolder::returnMomentsToCPU(CaloRecGPU::CUDA_Helpers::CUDAStreamPtrHolder stream)
{
  const cudaStream_t & stream_to_use = (stream != nullptr ? * ((cudaStream_t *) stream) : cudaStreamPerThread);
  cudaMemcpyAsync((CaloRecGPU::ClusterMomentsArr *) m_moments,
                  (CaloRecGPU::ClusterMomentsArr *) m_moments_dev,
                  sizeof(CaloRecGPU::ClusterMomentsArr),
                  cudaMemcpyDeviceToHost, stream_to_use);
}

void CaloRecGPU::EventDataHolder::returnClusterNumberToCPU(CaloRecGPU::CUDA_Helpers::CUDAStreamPtrHolder stream)
{
  const cudaStream_t & stream_to_use = (stream != nullptr ? * ((cudaStream_t *) stream) : cudaStreamPerThread);
  cudaMemcpyAsync(&(m_clusters->number), &(m_clusters_dev->number), sizeof(int), cudaMemcpyDeviceToHost, stream_to_use);
}


#define CALORECGPU_ASYNC_TRANSFER_HELPER(OBJ, MEMBER, TYPE, NUM, STREAM)            \
  CUDA_ERRCHECK( cudaMemcpyAsync ( &( OBJ -> MEMBER [0]),                           \
                                   &( OBJ ## _dev -> MEMBER [0]),                   \
                                   sizeof(TYPE) * (NUM),                            \
                                   cudaMemcpyDeviceToHost,                          \
                                   STREAM                           ) );

#define CALORECGPU_ASYNC_TRANSFER_PER_SAMPLE_HELPER(OBJ, MEMBER, TYPE, NUM, STREAM) \
  for (int sampling = 0; sampling < NumSamplings; ++sampling)                       \
    {                                                                               \
      CUDA_ERRCHECK( cudaMemcpyAsync ( &( OBJ -> MEMBER [sampling][0]),             \
                                       &( OBJ ## _dev -> MEMBER [sampling][0]),     \
                                       sizeof(TYPE) * (NUM),                        \
                                       cudaMemcpyDeviceToHost,                      \
                                       STREAM                       ) );            \
    }

//I almost felt tempted to do some
//variadic structured binding magic
//and compile time stuff
//to allow us to convert on-the-fly
//like this per-struct-member
//directly from the helpers.
//Emphasis on almost.
//Let's just hope for C++26...

void CaloRecGPU::EventDataHolder::returnSomeClustersToCPU(const size_t num_clusters, CaloRecGPU::CUDA_Helpers::CUDAStreamPtrHolder stream)
{
  const cudaStream_t & stream_to_use = (stream != nullptr ? * ((cudaStream_t *) stream) : cudaStreamPerThread);

  //We assume the cluster number we take is the known number of clusters,
  //so we skip copying that.

  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_clusters, clusterEnergy,      float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_clusters, clusterEt,          float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_clusters, clusterEta,         float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_clusters, clusterPhi,         float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_clusters, seedCellID,         int,   num_clusters, stream_to_use);
}

void CaloRecGPU::EventDataHolder::returnSomeMomentsToCPU(const size_t num_clusters, CaloRecGPU::CUDA_Helpers::CUDAStreamPtrHolder stream)
{
  const cudaStream_t & stream_to_use = (stream != nullptr ? * ((cudaStream_t *) stream) : cudaStreamPerThread);

  CALORECGPU_ASYNC_TRANSFER_PER_SAMPLE_HELPER(m_moments, energyPerSample,     float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_PER_SAMPLE_HELPER(m_moments, maxEPerSample,       float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_PER_SAMPLE_HELPER(m_moments, maxPhiPerSample,     float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_PER_SAMPLE_HELPER(m_moments, maxEtaPerSample,     float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_PER_SAMPLE_HELPER(m_moments, etaPerSample,        float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_PER_SAMPLE_HELPER(m_moments, phiPerSample,        float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, time,                float, num_clusters, stream_to_use);

  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, firstPhi,            float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, firstEta,            float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, secondR,             float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, secondLambda,        float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, deltaPhi,            float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, deltaTheta,          float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, deltaAlpha,          float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, centerX,             float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, centerY,             float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, centerZ,             float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, centerMag,           float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, centerLambda,        float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, lateral,             float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, longitudinal,        float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, engFracEM,           float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, engFracMax,          float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, engFracCore,         float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, firstEngDens,        float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, secondEngDens,       float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, isolation,           float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, engBadCells,         float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, nBadCells,           int,   num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, nBadCellsCorr,       int,   num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, badCellsCorrE,       float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, badLArQFrac,         float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, engPos,              float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, significance,        float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, cellSignificance,    float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, cellSigSampling,     int,   num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, avgLArQ,             float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, avgTileQ,            float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, engBadHVCells,       float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, nBadHVCells,         float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, PTD,                 float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, mass,                float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, EMProbability,       float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, hadWeight,           float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, OOCweight,           float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, DMweight,            float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, tileConfidenceLevel, float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, secondTime,          float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, nBadHVCells,         float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_PER_SAMPLE_HELPER(m_moments, nCellSampling,       int,   num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, nExtraCellSampling,  int,   num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, numCells,            int,   num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, vertexFraction,      float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, nVertexFraction,     float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, etaCaloFrame,        float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, etaCaloFrame,        float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, phiCaloFrame,        float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, eta1CaloFrame,       float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, phi1CaloFrame,       float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, eta2CaloFrame,       float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, phi2CaloFrame,       float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, engCalibTot,         float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, engCalibOutL,        float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, engCalibOutM,        float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, engCalibOutT,        float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, engCalibDeadL,       float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, engCalibDeadM,       float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, engCalibDeadT,       float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, engCalibDeadT,       float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, engCalibEMB0,        float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, engCalibEME0,        float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, engCalibTileG3,      float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, engCalibDeadTot,     float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, engCalibDeadEMB0,    float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, engCalibDeadTile0,   float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, engCalibDeadTileG3,  float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, engCalibDeadEME0,    float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, engCalibDeadHEC0,    float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, engCalibDeadFCAL,    float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, engCalibDeadLeakage, float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, engCalibDeadUnclass, float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, engCalibFracEM,      float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, engCalibFracHad,     float, num_clusters, stream_to_use);
  CALORECGPU_ASYNC_TRANSFER_HELPER(           m_moments, engCalibFracRest,    float, num_clusters, stream_to_use);
}

void CaloRecGPU::EventDataHolder::allocate(const bool also_GPU)
{
  m_cell_info.allocate();
  m_cell_state.allocate();
  m_clusters.allocate();
  m_moments.allocate();

  if (also_GPU)
    {
      m_cell_info_dev.allocate();
      m_cell_state_dev.allocate();
      m_clusters_dev.allocate();
      m_moments_dev.allocate();
    }
}

void CaloRecGPU::EventDataHolder::clear_GPU()
{
  m_cell_info_dev.clear();
  m_cell_state_dev.clear();
  m_clusters_dev.clear();
  m_moments_dev.clear();
}
