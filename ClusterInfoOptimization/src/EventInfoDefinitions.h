//
// Copyright (C) 2002-2023 CERN for the benefit of the ATLAS collaboration
//
// Dear emacs, this is -*- c++ -*-
//

#ifndef CALORECGPU_EVENTINFODEFINITIONS_H
#define CALORECGPU_EVENTINFODEFINITIONS_H

#include "BaseDefinitions.h"
#include "TagDefinitions.h"
#include "ConstantInfoDefinitions.h"

#include <cstdint>
#include <cmath>
//For fabsf...

#include "CaloSampling.h"
#include "LArNeighbours.h"

namespace CaloRecGPU
{

  /*! @class GainConversion
      Provides utility functions to handle the gain conversion.
  */
  class GainConversion
  {
    using GainType = unsigned int;
   protected:

    constexpr static GainType s_TileLowLow = 0,
                              s_TileLowHigh = 1,
                              s_TileHighLow = 2,
                              s_TileHighHigh = 3;

    constexpr static GainType s_TileOneLow = 0,
                              s_TileOneHigh = 3;
    //Are these valid/used?

    constexpr static GainType s_LArHigh = 0,
                              s_LArMedium = 1,
                              s_LArLow = 2;

    constexpr static GainType s_InvalidCell = 4;

    constexpr static GainType s_gain_values_minimum = 0;
    constexpr static GainType s_gain_values_maximum = 4;
   public:

    inline static constexpr GainType invalid_gain()
    {
      return s_InvalidCell;
    }

    inline static constexpr GainType max_gain_value()
    {
      return s_gain_values_maximum;
    }

    inline static constexpr GainType min_gain_value()
    {
      return s_gain_values_minimum;
    }

    inline static constexpr GainType num_gain_values()
    {
      return max_gain_value() - min_gain_value() + 1;
    }

    template <class T>
    inline static constexpr bool is_valid(const T & gain)
    {
      return gain != s_InvalidCell;
    }

    inline static constexpr GainType from_standard_gain(const int gain)
    //Basically, CaloCondUtils::getDbCaloGain without the Athena logging.
    {
      switch (gain)
        {
          case -16: //Tile LOWLOW
            return s_TileLowLow;
          case -15: //Tile LOWHIGH
            return s_TileLowHigh;
          case -12: //Tile HIGHLOW
            return s_TileHighLow;
          case -11: //Tile HIGHHIGH
            return s_TileHighHigh;
          case -4 : //Tile ONELOW
            return s_TileOneLow;
          case -3 : //Tile ONEHIGH
            return s_TileOneHigh;
          case 0  : //LAr High
            return s_LArHigh;
          case 1  : //LAr Medium
            return s_LArMedium;
          case 2  : //Lar Low
            return s_LArLow;
          default:
            return s_InvalidCell;
        }
    }

  };

  /*! @class QualityProvenance
      Just two uint16_t bit-packed onto a uint32_t.

      Not too worrisome for GPU performance since the kinds of things
      that ask the most for quality and provenance (cluster moments calculation)
      do ask for them both at the same time, so slower separate accesses
      are a non-issue.
  */
  struct QualityProvenance
  {
    using carrier = unsigned int;

    carrier value;

    constexpr static carrier s_16_bit_mask = 0xFFFFU;
    constexpr static carrier s_8_bit_mask = 0x00FFU;

   public:

    constexpr operator carrier () const
    {
      return value;
    }

    constexpr QualityProvenance (const carrier v): value(v)
    {
    }

    constexpr QualityProvenance & operator = (const carrier v)
    {
      value = v;
      return (*this);
    }

    constexpr unsigned int quality() const
    {
      return value & s_16_bit_mask;
    }

    constexpr unsigned int provenance() const
    {
      return (value >> 16) & s_16_bit_mask;
    }

    constexpr QualityProvenance(const uint16_t quality, const uint16_t provenance): value(provenance)
      //For non-tile
    {
      value = (value << 16) | quality;
    }

    constexpr QualityProvenance(const uint8_t q1, const uint8_t q2, const uint8_t q3, const uint8_t q4):
      value(q4)
      //For tile
    {
      value = (value << 8) | q3;
      value = (value << 8) | q2;
      value = (value << 8) | q1;
    }

    constexpr unsigned int tile_qual1() const
    {
      return value & s_8_bit_mask;
    }

    constexpr unsigned int tile_qual2() const
    {
      return (value >> 8) & s_8_bit_mask;
    }

    constexpr unsigned int tile_qbit1() const
    {
      return (value >> 16) & s_8_bit_mask;
    }

    constexpr unsigned int tile_qbit2() const
    {
      return (value >> 24) & s_8_bit_mask;
    }
  };

  struct CellInfoArr
  {
    float energy[NCaloCells];
    unsigned char gain[NCaloCells];
    float time[NCaloCells];
    QualityProvenance::carrier qualityProvenance[NCaloCells];
    //We could use/type pun a short2,
    //but let's go for portability for the time being...

    /*! GPU version of @c CaloBadCellHelper::isBad. If @p treat_L1_predicted_as_good is false,
        has the same effect as cell->badcell() (just like @c CaloBadCellHelper::isBad).
    */
    static constexpr bool is_bad(const bool is_tile, const QualityProvenance qp, const bool treat_L1_predicted_as_good = false)
    {
      bool ret = false;

      if (is_tile)
        {

          const unsigned int mask = 0x08U;

          ret = (qp.tile_qbit1() & mask) && (qp.tile_qbit2() & mask);
          //From TileCell::badcell()
          //badch1() && badch2()
          //badch1() -> m_tileQual[2]&TileCell::MASK_BADCH
          //badch2() -> m_tileQual[3]&TileCell::MASK_BADCH
          //
          //TileCell::MASK_BADCH= 0x08
          //
          //From CaloCell:
          //union {
          //  int  m_quality ;
          //  uint16_t m_qualProv[2];
          //  uint8_t m_tileQual[4];
          //};
          //quality returns m_qualProv[0]
          //provenance returns m_qualProv[1]
          //These are used to build our QualityProvenance...
          //
          //It's packing and unpacking back and forth, how fun!
        }
      else
        {
          const unsigned int provenance = qp.provenance();
          ret = provenance & 0x0800U;
          //As in LArCell::badcell()

          if (treat_L1_predicted_as_good && (provenance & 0x0200U))
            //As in CaloBadCellHelper::isBad
            {
              ret = false;
            }
        }
      return ret;
    }

    /*! GPU version of @c CaloBadCellHelper::isBad. If @p treat_L1_predicted_as_good is false,
        has the same effect as cell->badcell() (just like @c CaloBadCellHelper::isBad).
    */
    constexpr bool is_bad(const int cell, const bool treat_L1_predicted_as_good = false) const
    {
      return is_bad(GeometryArr::is_tile(cell), qualityProvenance[cell], treat_L1_predicted_as_good);
    }

    /*! GPU version of @c CaloBadCellHelper::isBad. If @p treat_L1_predicted_as_good is false,
        has the same effect as cell->badcell() (just like @c CaloBadCellHelper::isBad).
    */
    constexpr bool is_bad(const GeometryArr & geom, const int cell, const bool treat_L1_predicted_as_good = false) const
    {
      return is_bad(geom.is_tile(cell), qualityProvenance[cell], treat_L1_predicted_as_good);
    }

    /*! GPU equivalent of CaloTopoClusterMaker::passCellTimeCut.
    */
    constexpr bool passes_time_cut(const GeometryArr & geom, const int cell, const float threshold,
                                   const bool use_crosstalk, const float crosstalk_delta) const
    {
      const int sampling = geom.sampling(cell);
      if (sampling == CaloSampling::PreSamplerB ||
          sampling == CaloSampling::PreSamplerE ||
          sampling == CaloSampling::Unknown        )
        {
          return true;
        }
      else
        {
          const QualityProvenance qp = qualityProvenance[cell];
          const unsigned int mask = geom.is_tile(cell) ? 0x8080U : 0x2000U;
          if (qp.provenance() & mask)
            {
              const float this_time = time[cell];
              if (fabsf(this_time) < threshold)
                {
                  if (use_crosstalk)
                    {
                      const float this_energy = energy[cell];
                      const bool eligible = (sampling == CaloSampling::EMB2 || (sampling == CaloSampling::EME2 && fabsf(geom.eta[cell]) < 2.5));
                      if (this_energy > 0 && eligible)
                        {
                          int neighbours[NMaxNeighbours] = {};

                          constexpr unsigned int neigh_options = LArNeighbours::neighbourOption::prevInPhi | LArNeighbours::neighbourOption::nextInPhi;

                          const int num_neighs = geom.get_neighbours(neigh_options, cell, neighbours);

                          for (int i = 0; i < num_neighs; ++i)
                            {
                              const int neigh = neighbours[i];
                              if (is_valid(neigh) && energy[neigh] > 4 * this_energy)
                                {
                                  const QualityProvenance neigh_qp = qualityProvenance[neigh];
                                  if ( !(neigh_qp.provenance() & mask) || fabsf(time[neigh]) < threshold )
                                    {
                                      if (this_time > -threshold && this_time < threshold + crosstalk_delta)
                                        {
                                          return true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                  return false;
                }
            }
        }
      return true;
    }

    constexpr bool is_valid(const int cell) const
    {
      return GainConversion::is_valid(gain[cell]);
    }
  };

  struct CellStateArr
  {
    tag_type clusterTag[NCaloCells]; //cluster tag
  };

  struct ClusterInfoArr
  {
    int number;
    float clusterEnergy[NMaxClusters];
    float clusterEt[NMaxClusters];
    //Also used, as an intermediate value, to store AbsE
    float clusterEta[NMaxClusters];
    float clusterPhi[NMaxClusters];
    ///Invalid(ated) clusters have seedCellID < 0
    int seedCellID[NMaxClusters];
  };

  struct ClusterMomentsArr
//60 MB, but no explicit allocations needed.
//Worth the trade-off. GPU memory will only increase...
  {
    float energyPerSample     [NumSamplings][NMaxClusters];
    float maxEPerSample       [NumSamplings][NMaxClusters];
    float maxPhiPerSample     [NumSamplings][NMaxClusters];
    float maxEtaPerSample     [NumSamplings][NMaxClusters];
    float etaPerSample        [NumSamplings][NMaxClusters];
    float phiPerSample        [NumSamplings][NMaxClusters];
    float time                [NMaxClusters];
    //These are, strictly speaking, not moments,
    //but I think they are best left here rather than
    //in the ClusterInfoArr since they are only filled in
    //during cluster moments calculation...

    float firstPhi            [NMaxClusters];
    float firstEta            [NMaxClusters];
    float secondR             [NMaxClusters];
    float secondLambda        [NMaxClusters];
    float deltaPhi            [NMaxClusters];
    float deltaTheta          [NMaxClusters];
    float deltaAlpha          [NMaxClusters];
    float centerX             [NMaxClusters];
    float centerY             [NMaxClusters];
    float centerZ             [NMaxClusters];
    float centerMag           [NMaxClusters];
    float centerLambda        [NMaxClusters];
    float lateral             [NMaxClusters];
    float longitudinal        [NMaxClusters];
    float engFracEM           [NMaxClusters];
    float engFracMax          [NMaxClusters];
    float engFracCore         [NMaxClusters];
    float firstEngDens        [NMaxClusters];
    float secondEngDens       [NMaxClusters];
    float isolation           [NMaxClusters];
    float engBadCells         [NMaxClusters];
    int   nBadCells           [NMaxClusters];
    int   nBadCellsCorr       [NMaxClusters];
    float badCellsCorrE       [NMaxClusters];
    float badLArQFrac         [NMaxClusters];
    float engPos              [NMaxClusters];
    float significance        [NMaxClusters];
    float cellSignificance    [NMaxClusters];
    int   cellSigSampling     [NMaxClusters];
    float avgLArQ             [NMaxClusters];
    float avgTileQ            [NMaxClusters];
    float engBadHVCells       [NMaxClusters];
    float nBadHVCells         [NMaxClusters];
    float PTD                 [NMaxClusters];
    float mass                [NMaxClusters];
    float EMProbability       [NMaxClusters];
    float hadWeight           [NMaxClusters];
    float OOCweight           [NMaxClusters];
    float DMweight            [NMaxClusters];
    float tileConfidenceLevel [NMaxClusters];
    float secondTime          [NMaxClusters];
    int   nCellSampling       [NumSamplings][NMaxClusters];
    int   nExtraCellSampling  [NMaxClusters];
    int   numCells            [NMaxClusters];
    float vertexFraction      [NMaxClusters];
    float nVertexFraction     [NMaxClusters];
    float etaCaloFrame        [NMaxClusters];
    float phiCaloFrame        [NMaxClusters];
    float eta1CaloFrame       [NMaxClusters];
    float phi1CaloFrame       [NMaxClusters];
    float eta2CaloFrame       [NMaxClusters];
    float phi2CaloFrame       [NMaxClusters];
    float engCalibTot         [NMaxClusters];
    float engCalibOutL        [NMaxClusters];
    float engCalibOutM        [NMaxClusters];
    float engCalibOutT        [NMaxClusters];
    float engCalibDeadL       [NMaxClusters];
    float engCalibDeadM       [NMaxClusters];
    float engCalibDeadT       [NMaxClusters];
    float engCalibEMB0        [NMaxClusters];
    float engCalibEME0        [NMaxClusters];
    float engCalibTileG3      [NMaxClusters];
    float engCalibDeadTot     [NMaxClusters];
    float engCalibDeadEMB0    [NMaxClusters];
    float engCalibDeadTile0   [NMaxClusters];
    float engCalibDeadTileG3  [NMaxClusters];
    float engCalibDeadEME0    [NMaxClusters];
    float engCalibDeadHEC0    [NMaxClusters];
    float engCalibDeadFCAL    [NMaxClusters];
    float engCalibDeadLeakage [NMaxClusters];
    float engCalibDeadUnclass [NMaxClusters];
    float engCalibFracEM      [NMaxClusters];
    float engCalibFracHad     [NMaxClusters];
    float engCalibFracRest    [NMaxClusters];
    //And DigiHSTruth ones are reused here if that is the case?
    //Maybe counting from the end if we need to keep both?
  };

}

#endif