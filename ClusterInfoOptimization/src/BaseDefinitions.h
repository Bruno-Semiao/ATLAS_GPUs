//
// Copyright (C) 2002-2023 CERN for the benefit of the ATLAS collaboration
//
// Dear emacs, this is -*- c++ -*-
//

#ifndef CALORECGPU_BASEDEFINITIONS_H
#define CALORECGPU_BASEDEFINITIONS_H

namespace CaloRecGPU
{
  constexpr inline int NMaxNeighbours = 34;
  constexpr inline int NCaloCells = 187652;
  //Max index will be 0x0002 DD03
  
  constexpr inline int LArCellStart = 0;
  constexpr inline int TileCellStart = 182468;
  constexpr inline int LArCellAfterEnd = TileCellStart;
  constexpr inline int TileCellAfterEnd = NCaloCells;
  constexpr inline int NLArCells = LArCellAfterEnd - LArCellStart;
  constexpr inline int NTileCells = TileCellAfterEnd - TileCellStart;

  //These values should be constant
  //If there are any changes to that...
  //Well, we'll have to update them here.
  //But it's not something that would change at run time.
  
  constexpr inline int NMaxClusters = 0x10000U;
  //So it all fits in an int16_t
  //Would only start getting fishy if all the cells
  //were part of a cluster and all the clusters
  //had an average number of cells lower than 2.863...
  //(Or, in general, for N cells not part of a cluster,
  // an average number of cells per cluster lower than 2.863 - N/65536...)

  constexpr inline int NExactPairs = 2560816;
  constexpr inline int NMaxPairs = 0x400000U;
  //The actual (measured) value is 2560816, we rounded to the next power of two...

  constexpr inline int NumGainStates = 4;
  //This is the number of different gain states a cell can have.
  //(For the noise calculation.)
  
  constexpr inline int NumSamplings = 28;
  //The number of samplings in the calorimeter.
  
  constexpr inline int NumNeighOptions = 12;
  //The non-combined choices for LArNeighbours::neighbourOption:
  //prevInPhi      = 0x0001,  -> starts at 0
  //nextInPhi      = 0x0002,  -> starts at NeighOffset.get_number(0)
  //prevInEta      = 0x0004,  -> starts at NeighOffset.get_number(1)
  //nextInEta      = 0x0008,  -> starts at NeighOffset.get_number(2)
  //corners2D      = 0x0010,  -> starts at NeighOffset.get_number(3)
  //prevInSamp     = 0x0020,  -> starts at NeighOffset.get_number(4)
  //nextInSamp     = 0x0040,  -> starts at NeighOffset.get_number(5)
  //prevSubDet     = 0x0080,  -> starts at NeighOffset.get_number(6)
  //nextSubDet     = 0x0100,  -> starts at NeighOffset.get_number(7)
  //corners3D      = 0x0200,  -> starts at NeighOffset.get_number(8)
  //prevSuperCalo  = 0x0400,  -> starts at NeighOffset.get_number(9)
  //nextSuperCalo  = 0x0800   -> starts at NeighOffset.get_number(10)
  //            end           ->           NeighOffset.get_number(11)

}

#endif
