//
// Copyright (C) 2002-2023 CERN for the benefit of the ATLAS collaboration
//
// Dear emacs, this is -*- c++ -*-
//

#ifndef CALORECGPU_CONSTANTINFODEFINITIONS_H
#define CALORECGPU_CONSTANTINFODEFINITIONS_H

#include "BaseDefinitions.h"
#include "Helpers.h"

#include "EtaPhiMap.h"
#include "NeighArr.h"

namespace CaloRecGPU
{

  class ConstantEnumConversion
  {
    protected:
    
    constexpr static int s_subcalo_unknown             = 999999;
    constexpr static int s_subcalo_unknown_replacement = 7;
    
    public:
    
    constexpr static int from_subcalo_enum(const int subcalo)
    {
      if (subcalo == s_subcalo_unknown)
      {
        return s_subcalo_unknown_replacement;
      }
      else
      {
        return subcalo;
      }
    }
    
    constexpr static int to_subcalo_enum(const int subcalo)
    {
      if (subcalo == s_subcalo_unknown_replacement)
      {
        return s_subcalo_unknown;
      }
      else
      {
        return subcalo;
      }
    }
    
    protected:
    
    constexpr static int s_region_unknown             = 999999;
    constexpr static int s_region_unknown_replacement = 7;
    
    public:
    
    constexpr static int from_region_enum(const int region)
    {
      if (region == s_region_unknown)
      {
        return s_region_unknown_replacement;
      }
      else
      {
        return region;
      }
    }
    
    constexpr static int to_region_enum(const int region)
    {
      if (region == s_region_unknown_replacement)
      {
        return s_region_unknown;
      }
      else
      {
        return region;
      }
    }
    
    protected:
    
    constexpr static int s_sampling_unknown             = 999999;
    constexpr static int s_sampling_unknown_replacement = 7;
    
    public:
    
    constexpr static int from_intra_calorimeter_sampling_enum(const int sampling)
    {
      if (sampling == s_sampling_unknown)
      {
        return s_sampling_unknown_replacement;
      }
      else
      {
        return sampling;
      }
    }
    
    constexpr static int to_intra_calorimeter_sampling_enum(const int sampling)
    {
      if (sampling == s_sampling_unknown_replacement)
      {
        return s_sampling_unknown;
      }
      else
      {
        return sampling;
      }
    }
    
  };


/*! @class OtherCellInfo
    Packs the calo sampling, the intra-calorimeter sampling,
    the subcalo, the region and whether the cell should have
    its neighbours limited according to the PS and HEICW and FCal options.
    
    Possibly extensible with more information, as we have 16 bits still free...
*/
    struct OtherCellInfo
  {
    using carrier = unsigned int;

    carrier value;

private:

    static constexpr carrier s_sampling_mask       = 0x0000001FU;
    static constexpr carrier s_intra_sampling_mask = 0x000000E0U;
    static constexpr carrier s_subcalo_mask        = 0x00000700U;
    static constexpr carrier s_region_mask         = 0x00003800U;
    
    static constexpr carrier s_is_PS_flag          = 0x00004000U;
    static constexpr carrier s_is_HECIW_FCal_flag  = 0x00008000U;
    
    static constexpr carrier s_sampling_offset       =  0;
    static constexpr carrier s_intra_sampling_offset =  5;
    static constexpr carrier s_subcalo_offset        =  8;
    static constexpr carrier s_region_offset         = 11;
    
    static constexpr carrier s_bits_used       = s_sampling_mask       |
                                                 s_intra_sampling_mask |
                                                 s_subcalo_mask        |
                                                 s_region_mask         |
                                                 s_is_PS_flag          |
                                                 s_is_HECIW_FCal_flag;
    
    static constexpr carrier s_bits_unused     = ~s_bits_used;

public:

    constexpr operator carrier () const
    {
      return value;
    }

    constexpr OtherCellInfo (const carrier v): value(v)
    {
    }

    constexpr OtherCellInfo & operator = (const carrier v)
    {
      value = v;
      return (*this);
    }

    constexpr carrier sampling() const
    {
      return (value & s_sampling_mask) >> s_sampling_offset;
    }

    constexpr carrier intra_calorimeter_sampling() const
    {
      return (value & s_intra_sampling_mask) >> s_intra_sampling_offset;
    }
    
    constexpr carrier subcalo() const
    {
      return (value & s_subcalo_mask) >> s_subcalo_offset;
    }

    constexpr carrier region() const
    {
      return (value & s_region_mask) >> s_region_offset;
    }
    
    constexpr bool is_HECIW_or_FCal() const
    {
      return value & s_is_HECIW_FCal_flag;
    }
    
    constexpr bool is_PS() const
    {
      return value & s_is_PS_flag;
    }

    constexpr OtherCellInfo (const carrier sampling,
                             const carrier intra_calo_sampling,
                             const carrier subcalo,
                             const carrier region,
                             const bool HECIW_or_FCal,
                             const bool PS):
      value(0)
    {
      value  = ( sampling            << s_sampling_offset       ) |
               ( intra_calo_sampling << s_intra_sampling_offset ) |
               ( subcalo             << s_subcalo_offset        ) |
               ( region              << s_region_offset         ) |
               ( s_is_PS_flag * PS) | (s_is_HECIW_FCal_flag * HECIW_or_FCal);
    }
  };

  struct GeometryArr
  {
    float x[NCaloCells];
    float y[NCaloCells];
    float z[NCaloCells];
    float r[NCaloCells];
    float eta[NCaloCells];
    float phi[NCaloCells];

    float dx[NCaloCells];
    float dy[NCaloCells];
    float dz[NCaloCells];
    float dr[NCaloCells];
    float deta[NCaloCells];
    float dphi[NCaloCells];

    float volume[NCaloCells];

    NeighArr neighbours;
    NeighPairsArr neighPairs;

    EtaPhiToCellMap etaPhiToCell;

    OtherCellInfo::carrier otherCellInfo[NCaloCells];

    constexpr static bool is_tile (const int cell)
    {
      return cell >= TileCellStart && cell < TileCellAfterEnd;
    }
    
    constexpr bool is_HECIW_or_FCal(const int cell) const
    {
      const OtherCellInfo cell_info = otherCellInfo[cell];
      return cell_info.is_HECIW_or_FCal();
    }

    constexpr bool is_PS(const int cell) const
    {
      const OtherCellInfo cell_info = otherCellInfo[cell];
      return cell_info.is_PS();
    }
    
    constexpr int sampling(const int cell) const
    {
      const OtherCellInfo cell_info = otherCellInfo[cell];
      return cell_info.sampling(); 
    }
    
    constexpr int intra_calorimeter_sampling(const int cell) const
    {
      const OtherCellInfo cell_info = otherCellInfo[cell];
      return cell_info.intra_calorimeter_sampling(); 
    }
    
    constexpr int subcalo(const int cell) const
    {
      const OtherCellInfo cell_info = otherCellInfo[cell];
      return cell_info.subcalo(); 
    }
    
    constexpr int region(const int cell) const
    {
      const OtherCellInfo cell_info = otherCellInfo[cell];
      return cell_info.region(); 
    }
    
    /*! Places the neighbours according to the option(s) in neigh_options in the array
     *  and returns the number of such neighbours (not the total number of neighbours of the cell).
     *
     *  We're using C arrays for more immediate CUDA compatibility.
     */
    constexpr int get_neighbours(const unsigned int neigh_options, const int cell, int * neigh_arr) const
    {
      return neighbours.get_neighbours(neigh_options, cell, neigh_arr);
    }

    constexpr int get_number_of_neighbours(const unsigned int neigh_options, const int cell) const
    {
      return neighbours.get_number_of_neighbours(neigh_options, cell);
    }

    CUDA_HOS_DEV void fill_eta_phi_map()
    {
      for (int i = 0; i < NCaloCells; ++i)
        {
          etaPhiToCell.register_cell(i, sampling(i), eta[i], phi[i], deta[i], dphi[i]);
        }

      struct DeAllocWrapper
      {
        int * buf;
        CUDA_HOS_DEV DeAllocWrapper(const size_t new_size)
        {
          buf = new int[Helpers::int_ceil_div(new_size, sizeof(int))];
        }
        CUDA_HOS_DEV  ~DeAllocWrapper()
        {
          delete[] buf;
        }
        CUDA_HOS_DEV DeAllocWrapper (const DeAllocWrapper &) = delete;
        CUDA_HOS_DEV DeAllocWrapper (DeAllocWrapper &&) = delete;
        CUDA_HOS_DEV DeAllocWrapper & operator=(const DeAllocWrapper &) = delete;
        CUDA_HOS_DEV DeAllocWrapper & operator=(DeAllocWrapper &&) = delete;
      };
      //Simple allocation/de-allocation
      //to still be GPU-compatible if needed
      //(new and delete valid on GPU code...),
      //while still preventing leaks,
      //all without pulling in unique_ptr.

      DeAllocWrapper wrapper(etaPhiToCell.finish_initializing_buffer_size());

      etaPhiToCell.finish_initializing(wrapper.buf);

    }

    constexpr int get_closest_cell(const int sampling, const float test_eta, const float test_phi) const
    {
      int cells[EtaPhiToCellMap::s_max_overlap_cells] = {};

      const int n_cells = etaPhiToCell.get_possible_cells_from_coords(sampling, test_eta, test_phi, cells);

      if (n_cells < 1)
        {
          return -1;
        }
      else if (n_cells == 1)
        {
          return cells[0];
        }
      else
        {
          float distance = 1e38f;
          int ret = -1;

          for (int i = 0; i < n_cells; ++i)
            {
              const int this_cell = cells[i];

              const float delta_eta = eta[this_cell] - test_eta;
              const float delta_phi = Helpers::angular_difference(phi[this_cell], test_phi);

              const float this_dist = delta_eta * delta_eta + delta_phi * delta_phi;
              if (this_dist < distance || (this_dist == distance && this_cell > ret))
                {
                  distance = this_dist;
                  ret = this_cell;
                }
            }
          return ret;
        }
    }
  };

  struct CellNoiseProperties
  {
    using carrier = unsigned int;
    carrier value;

   protected:

    static constexpr carrier s_first_16_bit_mask = 0x0000FFFFU;
    static constexpr carrier s_last_16_bit_mask  = 0xFFFF0000U;
    static constexpr carrier s_invalid_double_gaussian = s_first_16_bit_mask | s_last_16_bit_mask;

   public:


    constexpr operator carrier () const
    {
      return value;
    }

    constexpr CellNoiseProperties (const carrier v): value(v)
    {
    }

    constexpr CellNoiseProperties & operator = (const carrier v)
    {
      value = v;
      return (*this);
    }

    constexpr carrier version() const
    {
      return value & s_first_16_bit_mask;
    }

    constexpr carrier noise_type() const
    {
      return (value & s_last_16_bit_mask) >> 16;
    }

    constexpr bool is_electronic_noise() const
    {
      return is_valid() && ((value & s_last_16_bit_mask) == 0x00000000U);
    }
    
    constexpr bool is_pile_up_noise() const
    {
      return is_valid() && ((value & s_last_16_bit_mask) == 0x00010000U);
    }

    constexpr bool is_total_noise() const
    {
      return is_valid() && ((value & s_last_16_bit_mask) == 0x00020000U);
    }
    
    constexpr CellNoiseProperties (const carrier version, const carrier noise_type):
      value(noise_type)
    {
      value = (value << 16) | (version & s_first_16_bit_mask);
    }

    constexpr bool is_invalid() const
    {
      return value == s_invalid_double_gaussian;
    }

    constexpr bool is_valid() const
    {
      return !is_invalid();
    }

    static constexpr carrier invalid_value()
    {
      return s_invalid_double_gaussian;
    }

  };

  struct CellNoiseArr
  {
    static constexpr int s_numDoubleGaussianConstants = 4;

    float noise[NCaloCells][NumGainStates];
    //Given the low number of possible gain sates

    float double_gaussian_constants[s_numDoubleGaussianConstants][NTileCells][NumGainStates];

    CellNoiseProperties::carrier noise_properties;

    float luminosity;

    constexpr float get_noise(const int cell, const int gain) const
    {
      return noise[cell][gain];
    }

   protected:

    CUDA_HOS_DEV float get_double_gaussian_significance(const int cell, const int gain, const float energy) const
    {

      using namespace std;

      const int delta_from_tile_start = cell - TileCellStart;

      const float sigma_1 = this->double_gaussian_constants[0][delta_from_tile_start][gain];
      const float sigma_2 = this->double_gaussian_constants[1][delta_from_tile_start][gain];
      const float ratio   = this->double_gaussian_constants[2][delta_from_tile_start][gain];
      
      if ((sigma_1 == 0.f && sigma_2 == 0.f) || energy == 0.f)
        {
          return 0.f;
        }
      else if (sigma_1 == 0.f)
        {
          return energy / sigma_2;
        }
      else if (ratio == 0.f || sigma_2 == 0.f)
        {
          return energy / sigma_1;
        }
        
      const float x_1 = energy / sigma_1;
      const float x_2 = energy / sigma_2;
      const float x_1_abs = fabsf(x_1);
      const float x_2_abs = fabsf(x_2);

      const float min_abs = min(x_1_abs, x_2_abs);

      const float max_abs = max(x_1_abs, x_2_abs);

      if (min_abs > 7.4f)
        {
          return min_abs;
        }
      if (max_abs < 0.9f)
        {
          return max_abs;
        }

      const float y_1 = erff(Helpers::Constants::inv_sqrt2<float> * x_1);
      const float y_2 = erff(Helpers::Constants::inv_sqrt2<float> * x_2);
      

      const float z = (y_1 * sigma_1 + ratio * y_2 * sigma_2) / (sigma_1 + ratio * sigma_2);
      
      const float ret = Helpers::Constants::sqrt2<float> * Helpers::erf_inv_wrapper(z);
      
      //printf("GPU %d %f %f %f %f %f %f\n", delta_from_tile_start, min_abs, max_abs, y_1, y_2, z, ret);
      
      return ret;

    }

   public:

    ///Calculates the double gaussian noise for a Tile cell.
    ///Does not check explicitly for a cell being from Tile.
    CUDA_HOS_DEV float get_double_gaussian_noise(const int cell, const int gain, const float energy) const
    {
      using namespace std;

      CellNoiseProperties props(this->noise_properties);
      if (!props.is_valid())
        {
          return get_noise(cell, gain);
        }

      const float sigma = get_double_gaussian_significance(cell, gain, energy);

      const float first_factor = (sigma != 0.f ? fabsf(energy / sigma) : 0.f);

      if (props.is_electronic_noise())
        {
          return first_factor;
        }
      else
        {
          const float second_factor = this->double_gaussian_constants[3][cell - TileCellStart][gain];

          switch (props.version())
            {
              case 1:
                return sqrtf(first_factor * first_factor + second_factor * second_factor * this->luminosity);
              case 2:
                return first_factor + second_factor * this->luminosity;
              default:
                return 0.f / 0.f;
            }
        }
    }

  };

}

#endif