//
// Copyright (C) 2002-2023 CERN for the benefit of the ATLAS collaboration
//
// Dear emacs, this is -*- c++ -*-
//

#ifndef CALORECGPU_NEIGHARR_H
#define CALORECGPU_NEIGHARR_H

#include "BaseDefinitions.h"

#include <climits>
//For CHAR_BIT


namespace CaloRecGPU
{

  /*! @class NeighOffset
      A class that expresses the offset from the beginning of the neighbours list
      for the several neighbour options.

    Bit packed to store all cumulative offsets,
    since they fit in 5 or 6 bits. (NMaxNeighbours == 34)
  */
  struct NeighOffset
  {
    using carrier = unsigned long long int;
    carrier value;

   protected:

    constexpr static carrier s_numbers_to_keep = NumNeighOptions - 1;
    //This is the number of cell offsets we actually need to keep.

    constexpr static int s_bits_per_offset = 5;
    constexpr static int s_bits_per_last_offset = 6;
    constexpr static int s_more_bits_begin = 9;
    //We start giving an extra bit to the offsets from
    //the 9th number (corresponding to prevSuperCalo)
    //since corners3D can push some cells to have more than 32 neighbours.


    constexpr static carrier s_offset_mask = 0x1FULL;
    constexpr static carrier s_offset_mask_last = 0x3FULL;
    constexpr static carrier s_offset_delta_pattern = 0x0208210842108421ULL;
    //This has one set bit every five,
    //except for the last three, that are one every six.

    constexpr static carrier s_last_bit_mask = 0x8000000000000000ULL;
    //This last bit is not currently used,
    //might be used as a flag if a use case comes up.

    constexpr static carrier s_only_numbers_mask = ~s_last_bit_mask;

    static_assert( s_more_bits_begin * s_bits_per_offset +
                   (s_numbers_to_keep - s_more_bits_begin) * s_bits_per_last_offset +
                   1 < sizeof(carrier) * CHAR_BIT, "All the offsets must fit within the carrier!" );

   public:

    constexpr operator carrier () const
    {
      return value;
    }

    constexpr NeighOffset (const carrier v): value(v)
    {
    }

    constexpr NeighOffset & operator = (const carrier v)
    {
      value = v;
      return (*this);
    }

    constexpr int get_number(const int i) const
    {
      carrier ret = value /*& s_only_numbers_mask*/;
      //Not necessary to mask since we mask the result anyway.

      if (i >= s_more_bits_begin)
        {
          ret = ret >> (s_bits_per_offset * s_more_bits_begin);
          ret = ret >> (s_bits_per_last_offset * (i - s_more_bits_begin));
          ret &= s_offset_mask_last;
        }
      else
        {
          ret = ret >> (s_bits_per_offset * i);
          ret &= s_offset_mask;
        }
      return (uint32_t) ret;
    }

    constexpr int get_start_cell (const int option) const
    {
      if (option == 0)
        {
          return 0;
        }
      /*else if (option < 0 || option >= NumNeighOptions)
        {
          return -1;
        }*/
      else
        {
          return get_number(option - 1);
        }
    }

    /*! To clarify, this cell no longer corresponds to the option,
        so it's one past the end (like the end iterator of a C++ container)...
    */
    constexpr int get_end_cell (const int option) const
    {
      /*
      if (option < 0 || option >= NumNeighOptions)
        {
          return -1;
        }
      else
        {*/
      return get_number(option);
      /*}*/
    }

    constexpr int get_total_number() const
    {
      return get_number(NumNeighOptions - 1);
    }

    constexpr int get_num_cells (const int option) const
    {
      return this->get_end_cell(option) - this->get_start_cell(option);
    }

    /*! Returns, for each @p i, what must be added to the offset
        to increase the stores numbers appropriately when a cell
        belonging to option @c (i-1) is added.

        In other words, the neighbour counts starting from option @p i are incremented.
    */
    static constexpr carrier offset_delta(const int i)
    {
      carrier mask = 0xFFFFFFFFFFFFFFFFULL;
      if (i >= s_more_bits_begin)
        {
          mask = mask << (s_bits_per_offset * s_more_bits_begin);
          mask = mask << (s_bits_per_last_offset * (i - s_more_bits_begin));
        }
      else
        {
          mask = mask << (s_bits_per_offset * i);
        }
      mask &= s_only_numbers_mask;
      return s_offset_delta_pattern & mask;
    }
  };

  struct NeighArr
  {
    NeighOffset::carrier offsets[NCaloCells];
    int cells[NCaloCells][NMaxNeighbours];
    //In most use cases so far,
    //we have to take neighbour options
    //into account, which makes this
    //much more performant
    //than the alternative...


    constexpr int get_total_number_of_neighbours(const int cell) const
    {
      const NeighOffset neigh_off = offsets[cell];
      return neigh_off.get_total_number();
    }

    constexpr int get_neighbour(const int cell, const int neigh_number) const
    {
      return cells[cell][neigh_number];
    }

    constexpr void set_neighbour(const int cell, const int neigh_number, const int neigh_v)
    {
      cells[cell][neigh_number] = neigh_v;
    }

    /*! Places the neighbours according to the option(s) in neigh_options in the array
     *  and returns the number of such neighbours (not the total number of neighbours of the cell).
     *
     *  We're using C arrays for more immediate CUDA compatibility.
     */
    constexpr int get_neighbours(const unsigned int neigh_options, const int cell, int * neigh_arr) const
    {
      const NeighOffset neigh_off = offsets[cell];

      int neigh_arr_len = 0;
      int limit = 0;
      int neigh = 0;

      for (int i = 0; i < NumNeighOptions; ++i)
        {
          limit = neigh_off.get_end_cell(i);
          if ( neigh_options & (1U << i) )
            {
              for (; neigh < limit; ++neigh)
                {
                  neigh_arr[neigh_arr_len] = cells[cell][neigh];
                  ++neigh_arr_len;
                }
            }
          neigh = limit;
        }
      return neigh_arr_len;
    }

    constexpr int get_number_of_neighbours(const unsigned int neigh_options, const int cell) const
    {
      const NeighOffset neigh_off = offsets[cell];
      int ret = 0;
      for (int i = 0; i < NumNeighOptions; ++i)
        {
          if ( neigh_options & (1U << i) )
            {
              ret += neigh_off.get_num_cells(i);
            }
        }
      return ret;
    }
  };

  struct NeighPairsArr
  {
    //const int NExactPairs = 2560816;
    int cell_A[NExactPairs];
    int cell_B[NExactPairs];

    constexpr int get_option(const int pair) const
    {
      constexpr int s_pairs_end[NumNeighOptions] = {184128, 368256, 551424, 733312, 1483056, 1960492, 2437928, 2438824, 2439720, 2500008, 2530412, 2560816};
      for (int i = 0; i < NumNeighOptions; ++i)
        {
          if (pair < s_pairs_end[i])
            {
              return i;
            }
        }
      return -1;
    }

    constexpr bool is_valid(const int pair, const unsigned int options, const bool limited_PS, const bool limited_HECIW_FCal) const
    {
      const int opt_idx = get_option(pair);
      if (opt_idx < 0 || !(options & (1U << opt_idx)) )
        {
          return false;
        }

      constexpr int limited_option = 6;
      //nextInSampl


      constexpr int s_PS_start[NumNeighOptions] = {173888, 358016, 541056, 723584, 1432784, 1954480, 2364410, 2438696, 2439336, 2500008, 2529160, 2559756};
      constexpr int s_HECIW_FCal_start[NumNeighOptions] = {183232, 367360, 550272, 732672, 1469392, 1954480, 2435450, 2438824, 2439720, 2500008, 2529160, 2559756};
      
      if (limited_PS && opt_idx != limited_option)
        {
          if (pair >= s_PS_start[opt_idx] && pair < s_HECIW_FCal_start[opt_idx])
            {
              return false;
            }
        }

      if (limited_HECIW_FCal && opt_idx != limited_option)
        {
          if (pair >= s_HECIW_FCal_start[opt_idx])
            {
              return false;
            }
        }

      return true;

    }

  };

}

#endif //CALORECGPU_NEIGHARR_H