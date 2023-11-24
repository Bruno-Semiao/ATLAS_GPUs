//
// Copyright (C) 2002-2023 CERN for the benefit of the ATLAS collaboration
//
// Dear emacs, this is -*- c++ -*-
//

#ifndef CALORECGPU_IGPUKERNELSIZEOPTIMIZER_H
#define CALORECGPU_IGPUKERNELSIZEOPTIMIZER_H

#include <string>

struct CUDAKernelLaunchConfiguration
{
  int grid_x = 0, grid_y = 0, grid_z = 0, block_x = 0, block_y = 0, block_z = 0;
}KernelConfig;

/**
 * @class  IGPUKernelSizeOptimizer
 * @author Nuno Fernandes <nuno.dos.santos.fernandes@cern.ch>
 * @date 03 August 2023
 * @brief Interface for GPU kernel size optimization
 * (allowing adjustment of kernel sizes to the properties of the available device).
 *
 * This class should remain independent of any Athena-centric includes,
 * so that it can be straightforwardly used from within .cu files.
 */

class IGPUKernelSizeOptimizer
{
 public:

  enum SpecialSizeHints
  {
    CooperativeLaunch = -1
  };

  /** @brief Register a kernel with a specific name.  */
  
  virtual void register_kernel(const std::string & kernel_name,
                               void * kernel,
                               const int blocksize_hint,
                               const int gridsize_hint,
                               const int max_total_threads)
  {
    this->register_kernels(kernel_name, 1, &kernel, &blocksize_hint, &gridsize_hint, &max_total_threads, 0);
  }

  /** @brief Register a set of kernels that can be referred back to with a name and a number.

      Uses C-style arrays for more immediate CUDA compatibility,
      assumes the size of @p kernels, @p blocksize_hints and @p gridsize_hints is @p number,
      and starts the numbering with an optional @p offset.
  */
  virtual void register_kernels(const std::string & tool_name,
                                const int number,
                                void ** kernels,
                                const int * blocksize_hints,
                                const int * gridsize_hints,
                                const int * max_total_threads,
                                const int offset = 0) = 0;

  /** @brief Retrieve the (hopefully optimal) kernel launch configuration.*/
  virtual CUDAKernelLaunchConfiguration get_launch_configuration(const std::string & name,
                                                                 const int number = 0,
                                                                 const int dynamic_memory = 0) const = 0;

  /** @brief Whether the device + environment in use support cooperative groups. */
  virtual bool can_use_cooperative_groups() const = 0;

  /** @brief Whether the device + environment in use support dynamic parallelism. */
  virtual bool can_use_dynamic_parallelism() const = 0;

  /** @brief Whether to avoid oversizing kernels and instead (if possible) launch kernels with the exact number of threads... */
  virtual bool use_minimal_kernel_sizes() const
  {
    //Testing shows that, at least on the devices we use,
    //we only lose performance by dyn-par'ing our way to do this.
    return false;
  }

  /** @brief Set grid and block sizes for optimization. */
  virtual void set_kernel_sizes (const int grid, const int block)
  {
    gridSize  = grid;
    blockSize = block;
  }

  /** @brief Get current grid and block sizes*/
  virtual int get_kernel_sizes (const char opt = 'g') const //g = grid; b = block
  {
    if(opt=='g')      return gridSize;
    else if (opt=='b')return blockSize;
    else              return 0;
  }

  virtual ~IGPUKernelSizeOptimizer() = default;

 protected:
  int gridSize = 32, blockSize = 32;
};
#endif