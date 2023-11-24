//
// Copyright (C) 2002-2023 CERN for the benefit of the ATLAS collaboration
//
// Dear emacs, this is -*- c++ -*-
//

#ifndef CALORECGPU_TAGDEFINITIONS_H
#define CALORECGPU_TAGDEFINITIONS_H

#include "BaseDefinitions.h"

#include <cstdint>

namespace CaloRecGPU
{
  /*! @class TagBase
      The base class for marking cells as belonging to clusters.

      Currently 64-bit based:
      - Most significant bit is a flag
      - Next 16 bits are meant for a `uint16_t`
      - Next 31 bits are meant to hold the bit pattern of
      a non-negative `float` (sign bit is zero, so we ignore it)
      - Next 16 bits are meant for another `uint16_t`

      The actual arrays are intended to be
      of @c TagBase::carrier rather than @c TagBase,
      to keep everything as native types on the GPU side.

  */
  struct TagBase
  {
   public:

    using carrier = unsigned long long int;
    //uint64_t is unsigned long long int in CUDA.

    carrier value;

    constexpr operator carrier () const
    {
      return value;
    }

    constexpr TagBase (const carrier v): value(v)
    {
    }

    constexpr TagBase & operator = (const carrier v)
    {
      value = v;
      return (*this);
    }

   protected:

    constexpr static carrier s_bit_mask = 0x8000000000000000ULL;
    //0x 8000 0000 0000 0000

    constexpr static carrier s_second_16bit_mask = 0x7FFF800000000000ULL;
    //0x 7FFF 8000 0000 0000

    constexpr static carrier s_middle_31bit_mask = 0x00007FFFFFFF0000ULL;
    //0x 0000 7FFF FFFF 0000

    constexpr static carrier s_first_16bit_mask =  0x000000000000FFFFULL;
    //0x 0000 0000 0000 FFFF

   public:

    [[nodiscard]] constexpr bool get_flag() const
    {
      return value & s_bit_mask;
    }

    [[nodiscard]] constexpr uint32_t get_second_16bit() const
    //We use ints to better fit CUDA hardware.
    {
      carrier pre_shift = value & s_second_16bit_mask;
      uint32_t ret = pre_shift >> 47;
      return ret;
    }

    [[nodiscard]] constexpr uint32_t get_middle_31bit() const
    {
      carrier pre_shift = value & s_middle_31bit_mask;
      uint32_t ret = pre_shift >> 16;
      return ret;
    }

    [[nodiscard]] constexpr uint32_t get_first_16bit() const
    //We use ints to better fit CUDA hardware.
    {
      uint32_t ret = value & s_first_16bit_mask;
      return ret;
    }

    [[nodiscard]] constexpr carrier clear_flag() const
    {
      return value & (~ s_bit_mask);
    }

    [[nodiscard]] constexpr carrier clear_second_16bit() const
    {
      return value & (~ s_second_16bit_mask);
    }

    [[nodiscard]] constexpr carrier clear_middle_31bit() const
    {
      return value & (~ s_middle_31bit_mask);
    }

    [[nodiscard]] constexpr carrier clear_first_16bit() const
    {
      return value & (~ s_first_16bit_mask);
    }

    [[nodiscard]] constexpr carrier keep_only_flag() const
    {
      return value & s_bit_mask;
    }

    [[nodiscard]] constexpr carrier keep_only_second_16bit() const
    {
      return value & s_second_16bit_mask;
    }

    [[nodiscard]] constexpr carrier keep_only_middle_31bit() const
    {
      return value & s_middle_31bit_mask;
    }

    [[nodiscard]] constexpr carrier keep_only_first_16bit() const
    {
      return value & s_first_16bit_mask;
    }

    [[nodiscard]] constexpr carrier or_flag(const bool bit) const
    {
      return value | (bit * s_bit_mask);
    }

    [[nodiscard]] constexpr carrier or_second_16bit(const uint16_t pattern) const
    //We use ints to better fit CUDA hardware.
    {
      carrier temp = pattern;
      return value | (temp << 47);
    }

    [[nodiscard]] constexpr carrier or_middle_31bit(const uint32_t pattern) const
    {
      carrier temp = pattern & 0x7FFFFFFFU;
      //To make sure the highest bit is unset.
      return value | (temp << 16);
    }

    [[nodiscard]] constexpr carrier or_first_16bit(const uint16_t pattern) const
    //We use ints to better fit CUDA hardware.
    {
      uint32_t temp = pattern;
      return value | temp;
    }

    [[nodiscard]] constexpr carrier override_flag(const bool bit) const
    {
      return (value & (~ s_bit_mask)) | (bit * s_bit_mask);
    }

    [[nodiscard]] constexpr carrier override_second_16bit(const uint16_t pattern) const
    //We use ints to better fit CUDA hardware.
    {
      const carrier temp = pattern;
      return (value & (~ s_second_16bit_mask)) | (temp << 47);
    }

    [[nodiscard]] constexpr carrier override_middle_31bit(const uint32_t pattern) const
    {
      const carrier temp = pattern & 0x7FFFFFFFU;
      //To make sure the highest bit is unset.
      return (value & (~ s_middle_31bit_mask)) | (temp << 16);
    }

    [[nodiscard]] constexpr carrier override_first_16bit(const uint16_t pattern) const
    //We use ints to better fit CUDA hardware.
    {
      const uint32_t temp = pattern;
      return (value & (~ s_first_16bit_mask)) | temp;
    }

    [[nodiscard]] static constexpr carrier make_base_tag(const uint16_t lower = 0, const uint32_t middle = 0, const uint16_t upper = 0, const bool last = false)
    {
      carrier ret = upper;
      ret = (ret << 31) | (middle & 0x7FFFFFFFU);
      ret = (ret << 16) | lower;

      ret = ret | (s_bit_mask * last);

      return ret;
    }

    [[nodiscard]] static constexpr carrier make_invalid_tag()
    {
      return 0ULL;
    }

  };

  /*! @class ClusterTag
      The class that actually expresses the cluster assignment.
      Supports cells that are shared between two clusters,
      storing the weight assigned to the first
      (as the weight assigned to the second will be one minus that);
      cells that are part of a single cluster
      have that weight set to
      (the bit pattern of the floating point representation of) 1.

      See @c TagBase for more details.


  */
  struct ClusterTag : public TagBase
  {
   public:

    using TagBase::TagBase;


    [[nodiscard]] static constexpr carrier make_tag(const uint16_t cluster_index = 0, const int32_t weight = 0, const uint16_t second_cluster_index = 0)
    {
      return make_base_tag(cluster_index, weight, second_cluster_index, true);
    }

    [[nodiscard]] constexpr bool is_part_of_cluster() const
    {
      return this->get_flag();
    }

    [[nodiscard]] static constexpr bool is_part_of_cluster(const ClusterTag tag)
    {
      return tag.is_part_of_cluster();
    }

    [[nodiscard]] constexpr int32_t cluster_index() const
    {
      return this->get_first_16bit();
    }

    [[nodiscard]] static constexpr int32_t cluster_index(const ClusterTag tag)
    {
      return tag.cluster_index();
    }

    [[nodiscard]] constexpr int32_t secondary_cluster_index() const
    {
      return this->get_second_16bit();
    }

    [[nodiscard]] static constexpr int32_t secondary_cluster_index(const ClusterTag tag)
    {
      return tag.secondary_cluster_index();
    }

    [[nodiscard]] constexpr int32_t secondary_cluster_weight() const
    {
      return this->get_middle_31bit();
    }

    [[nodiscard]] static constexpr int32_t secondary_cluster_weight(const ClusterTag tag)
    {
      return tag.secondary_cluster_weight();
    }

    [[nodiscard]] constexpr bool is_shared_between_clusters() const
    {
      return this->is_part_of_cluster() && this->secondary_cluster_weight() > 0;
    }

    [[nodiscard]] static constexpr bool is_shared_between_clusters(const ClusterTag tag)
    {
      return tag.is_shared_between_clusters();
    }

    [[nodiscard]] constexpr carrier set_part_of_cluster(const bool is_part = true) const
    {
      return this->or_flag(is_part);
    }

    [[nodiscard]] constexpr carrier set_cluster_index(const int32_t index) const
    {
      return this->or_first_16bit(index & 0xFFFFU);
    }

    [[nodiscard]] constexpr carrier set_secondary_cluster_index(const int32_t index) const
    {
      return this->or_second_16bit(index & 0xFFFFU);
    }

    [[nodiscard]] constexpr carrier set_secondary_cluster_weight(const int32_t weight) const
    {
      return this->or_middle_31bit(weight);
    }

    [[nodiscard]] constexpr carrier override_part_of_cluster(const bool is_part = true) const
    {

      return this->override_flag(is_part);
    }

    [[nodiscard]] constexpr carrier override_cluster_index(const int32_t index) const
    {
      return this->override_first_16bit(index & 0xFFFFU);
    }

    [[nodiscard]] constexpr carrier override_secondary_cluster_index(const int32_t index) const
    {
      return this->override_second_16bit(index & 0xFFFFU);
    }

    [[nodiscard]] constexpr carrier override_secondary_cluster_weight(const int32_t weight) const
    {
      return this->override_middle_31bit(weight);
    }
  };

  using tag_type = TagBase::carrier;
}

#endif
