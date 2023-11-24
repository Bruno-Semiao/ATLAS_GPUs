//
// Copyright (C) 2002-2023 CERN for the benefit of the ATLAS collaboration
//
// Dear emacs, this is -*- c++ -*-
//

#ifndef CALORECGPU_STANDALONEDATAIO_H
#define CALORECGPU_STANDALONEDATAIO_H

#include "Helpers.h"

#include "CUDAFriendlyClasses.h"

#include <fstream>
#include <string>
#include <map>
#include <set>
#include <iomanip>

#include <filesystem>

struct StandaloneDataIO
{
  enum class ErrorState
  {
    OK = 0, ReadError, WriteError
  };

 protected:

  inline static void report_error(const std::filesystem::path & file, const std::string & kind, const bool really_report = false)
  {
    if (really_report)
      {
        std::cerr << "ERROR: when " << kind << " from '" << file << "'." << std::endl;
      }
  }

 public:

  struct ConstantInformation
  {
    friend class StandaloneDataIO;

   public:

    inline static ErrorState read_geometry(const std::filesystem::path & file,
                                           CaloRecGPU::Helpers::CPU_object<CaloRecGPU::GeometryArr> & geo,
                                           const bool report = false)
    {
      std::ifstream in(file.native(), std::ios_base::binary);
      geo.binary_input(in);
      if (in.fail())
        {
          report_error(file, "reading geometry", report);
          return ErrorState::ReadError;
        }
      in.close();
      return ErrorState::OK;
    }

    inline static ErrorState read_noise(const std::filesystem::path & file,
                                        CaloRecGPU::Helpers::CPU_object<CaloRecGPU::CellNoiseArr> & noise,
                                        const bool report = false)
    {
      std::ifstream in(file.native(), std::ios_base::binary);
      noise.binary_input(in);
      if (in.fail())
        {
          report_error(file, "reading noise", report);
          return ErrorState::ReadError;
        }
      in.close();
      return ErrorState::OK;
    }

    inline static ErrorState write_geometry(std::filesystem::path file,
                                            const CaloRecGPU::Helpers::CPU_object<CaloRecGPU::GeometryArr> & geo,
                                            const bool report = false)
    {
      file.replace_extension(".geometry");
      std::ofstream out(file, std::ios_base::binary);
      geo.binary_output(out);
      if (out.fail())
        {
          report_error(file, "writing geometry", report);
          return ErrorState::WriteError;
        }
      out.close();
      return ErrorState::OK;
    }

    inline static ErrorState write_noise(std::filesystem::path file,
                                         const CaloRecGPU::Helpers::CPU_object<CaloRecGPU::CellNoiseArr> & noise,
                                         const bool report = false)
    {
      file.replace_extension(".noise");
      std::ofstream out(file, std::ios_base::binary);
      noise.binary_output(out);
      if (out.fail())
        {
          report_error(file, "writing noise", report);
          return ErrorState::WriteError;
        }
      out.close();
      return ErrorState::OK;
    }
  };

  struct EventInformation
  {
    friend class StandaloneDataIO;

   public:
    inline static ErrorState read_cluster_info(const std::filesystem::path & file,
                                               CaloRecGPU::Helpers::CPU_object<CaloRecGPU::ClusterInfoArr> & clusters,
                                               const bool report = false)
    {
      clusters.allocate();

      std::ifstream in(file.native(), std::ios_base::binary);

      in.read((char *) & (clusters->number), sizeof(int));

      if (in.fail() || clusters->number < 0 || clusters->number > CaloRecGPU::NMaxClusters)
        {
          report_error(file, "reading clusters", report);
          return ErrorState::ReadError;
        }

      in.read((char *) clusters->clusterEnergy, sizeof(float) * clusters->number);
      in.read((char *) clusters->clusterEt, sizeof(float) * clusters->number);
      in.read((char *) clusters->clusterEta, sizeof(float) * clusters->number);
      in.read((char *) clusters->clusterPhi, sizeof(float) * clusters->number);
      in.read((char *) clusters->seedCellID, sizeof(int) * clusters->number);

      if (in.fail())
        {
          report_error(file, "reading clusters", report);
          return ErrorState::ReadError;
        }

      in.close();

      return ErrorState::OK;

    }

    inline static ErrorState read_cell_info(const std::filesystem::path & file,
                                            CaloRecGPU::Helpers::CPU_object<CaloRecGPU::CellInfoArr> & cell_info,
                                            const bool report = false)
    {
      std::ifstream in(file.native(), std::ios_base::binary);
      cell_info.binary_input(in);
      if (in.fail())
        {
          report_error(file, "reading cell info", report);
          return ErrorState::ReadError;
        }
      in.close();
      return ErrorState::OK;
    }

    inline static ErrorState read_cell_state(const std::filesystem::path & file,
                                             CaloRecGPU::Helpers::CPU_object<CaloRecGPU::CellStateArr> & cell_state,
                                             const bool report = false)
    {
      std::ifstream in(file.native(), std::ios_base::binary);
      cell_state.binary_input(in);
      if (in.fail())
        {
          report_error(file, "reading cell state", report);
          return ErrorState::ReadError;
        }
      in.close();
      return ErrorState::OK;
    }

    inline static ErrorState write_cluster_info(std::filesystem::path file,
                                                const CaloRecGPU::Helpers::CPU_object<CaloRecGPU::ClusterInfoArr> & clusters,
                                                const bool report = false)
    {
      file.replace_extension(".clusterinfo");
      std::ofstream out(file, std::ios_base::binary);

      out.write((char *) & (clusters->number), sizeof(int));
      out.write((char *) clusters->clusterEnergy, sizeof(float) * clusters->number);
      out.write((char *) clusters->clusterEt, sizeof(float) * clusters->number);
      out.write((char *) clusters->clusterEta, sizeof(float) * clusters->number);
      out.write((char *) clusters->clusterPhi, sizeof(float) * clusters->number);
      out.write((char *) clusters->seedCellID, sizeof(int) * clusters->number);

      if (out.fail())
        {
          report_error(file, "writing clusters", report);
          return ErrorState::WriteError;
        }
      out.close();
      return ErrorState::OK;
    }

    inline static ErrorState write_cell_info(std::filesystem::path file,
                                             const CaloRecGPU::Helpers::CPU_object<CaloRecGPU::CellInfoArr> & cell_info,
                                             const bool report = false)
    {
      file.replace_extension(".cellinfo");
      std::ofstream out(file, std::ios_base::binary);
      cell_info.binary_output(out);
      if (out.fail())
        {
          report_error(file, "writing cell info", report);
          return ErrorState::WriteError;
        }
      out.close();
      return ErrorState::OK;
    }

    inline static ErrorState write_cell_state(std::filesystem::path file,
                                              const CaloRecGPU::Helpers::CPU_object<CaloRecGPU::CellStateArr> & cell_state,
                                           const bool report = false)
    {
      file.replace_extension(".cellstate");
      std::ofstream out(file, std::ios_base::binary);
      cell_state.binary_output(out);
      if (out.fail())
        {
          report_error(file, "writing cell state", report);
          return ErrorState::WriteError;
        }
      out.close();
      return ErrorState::OK;
    }

  };

 protected:

  inline static bool create_or_check_folder(const std::filesystem::path & folder, const bool output_errors = true)
  {
    if (!std::filesystem::exists(folder))
      {
        if (!std::filesystem::create_directory(folder))
          {
            if (output_errors)
              {
                std::cout << "ERROR: folder '" << folder << "' could not be created." << std::endl;
              }
            return false;
          }
      }
    else if (!std::filesystem::is_directory(folder))
      {
        if (output_errors)
          {
            std::cout << "ERROR: folder '" << folder << "' is not a valid folder." << std::endl;
          }
        return false;
      }
    return true;
  }

 public:

  inline static ErrorState prepare_folder_for_output(const std::filesystem::path & folder, const bool output_errors = true)
  {
    if (!create_or_check_folder(folder, output_errors))
      {
        return ErrorState::WriteError;
      }
    return ErrorState::OK;
  }

  inline static std::string build_filename( const std::string & prefix,
                                            const std::string & text,
                                            const std::string & suffix,
                                            const std::string & ext      )
  {
    return prefix + (prefix.size() > 0 ? "_" : "") + text +
           (suffix.size() > 0 ? "_" : "") + suffix + "." + ext;
  }

  inline static std::string build_filename( const std::string & prefix,
                                            const size_t event_number,
                                            const std::string & suffix,
                                            const std::string & ext,
                                            const unsigned int num_width = 9 )
  {
    std::ostringstream event_ID_format;
    event_ID_format << std::setfill('0') << std::setw(num_width) << event_number;
    const std::string event_ID = event_ID_format.str();
    return build_filename(prefix, event_ID, suffix, ext);
  }


  inline static ErrorState save_constants_to_folder(const std::filesystem::path & folder,
                                                    const CaloRecGPU::Helpers::CPU_object<CaloRecGPU::GeometryArr> & geo,
                                                    const CaloRecGPU::Helpers::CPU_object<CaloRecGPU::CellNoiseArr> & noise,
                                                    const std::string & prefix = "",
                                                    const std::string & suffix = "",
                                                    const bool output_errors = true)
  {
    if (!create_or_check_folder(folder, output_errors))
      {
        return ErrorState::WriteError;
      }

    auto filename = [&] (const std::string & text, const std::string & ext)
    {
      return folder / build_filename(prefix, text, suffix, ext);
    };

    if (ConstantInformation::write_geometry(filename("geometry", "geo"), geo) != ErrorState::OK)
      {
        return ErrorState::WriteError;
      }
    if (ConstantInformation::write_noise(filename("noise", "noise"), noise) != ErrorState::OK)
      {
        return ErrorState::WriteError;
      }
    return ErrorState::OK;
  }

  inline static ErrorState save_event_to_folder(const size_t event_number,
                                                const std::filesystem::path & folder,
                                                const CaloRecGPU::Helpers::CPU_object<CaloRecGPU::CellInfoArr> & cell_info,
                                                const CaloRecGPU::Helpers::CPU_object<CaloRecGPU::CellStateArr> & cell_state,
                                                const CaloRecGPU::Helpers::CPU_object<CaloRecGPU::ClusterInfoArr> & clusters,
                                                const std::string & prefix = "",
                                                const std::string & suffix = "",
                                                const unsigned int num_width = 9,
                                                const bool output_errors = true)
  {
    if (!create_or_check_folder(folder, output_errors))
      {
        return ErrorState::WriteError;
      }

    std::ostringstream event_ID_format;
    event_ID_format << std::setfill('0') << std::setw(num_width) << event_number;
    const std::string event_ID = event_ID_format.str();

    auto filename = [&] (const std::string & ext)
    {
      return folder / build_filename(prefix, event_ID, suffix, ext);
    };

    if (EventInformation::write_cell_info(filename("cellinfo"), cell_info) != ErrorState::OK)
      {
        return ErrorState::WriteError;
      }
    if (EventInformation::write_cell_state(filename("cellstate"), cell_state) != ErrorState::OK)
      {
        return ErrorState::WriteError;
      }
    if (EventInformation::write_cluster_info(filename("clusterinfo"), clusters) != ErrorState::OK)
      {
        return ErrorState::WriteError;
      }
    return ErrorState::OK;
  }

  inline static ErrorState save_cell_state_to_folder(const size_t event_number,
                                                     const std::filesystem::path & folder,
                                                     const CaloRecGPU::Helpers::CPU_object<CaloRecGPU::CellStateArr> & cell_state,
                                                     const std::string & prefix = "",
                                                     const std::string & suffix = "",
                                                     const unsigned int num_width = 9,
                                                     const bool output_errors = true)
  {
    if (!create_or_check_folder(folder, output_errors))
      {
        return ErrorState::WriteError;
      }

    std::ostringstream event_ID_format;
    event_ID_format << std::setfill('0') << std::setw(num_width) << event_number;
    const std::string event_ID = event_ID_format.str();

    auto filename = [&] (const std::string & ext)
    {
      return folder / build_filename(prefix, event_ID, suffix, ext);
    };

    if (EventInformation::write_cell_state(filename("cellstate"), cell_state) != ErrorState::OK)
      {
        return ErrorState::WriteError;
      }
    return ErrorState::OK;
  }

  inline static ErrorState save_cell_info_to_folder(const size_t event_number,
                                                    const std::filesystem::path & folder,
                                                    const CaloRecGPU::Helpers::CPU_object<CaloRecGPU::CellInfoArr> & cell_info,
                                                    const std::string & prefix = "",
                                                    const std::string & suffix = "",
                                                    const unsigned int num_width = 9,
                                                    const bool output_errors = true)
  {
    if (!create_or_check_folder(folder, output_errors))
      {
        return ErrorState::WriteError;
      }

    std::ostringstream event_ID_format;
    event_ID_format << std::setfill('0') << std::setw(num_width) << event_number;
    const std::string event_ID = event_ID_format.str();

    auto filename = [&] (const std::string & ext)
    {
      return folder / build_filename(prefix, event_ID, suffix, ext);
    };

    if (EventInformation::write_cell_info(filename("cellinfo"), cell_info) != ErrorState::OK)
      {
        return ErrorState::WriteError;
      }
    return ErrorState::OK;
  }

  inline static ErrorState save_clusters_to_folder(const size_t event_number,
                                                   const std::filesystem::path & folder,
                                                   const CaloRecGPU::Helpers::CPU_object<CaloRecGPU::ClusterInfoArr> & clusters,
                                                   const std::string & prefix = "",
                                                   const std::string & suffix = "",
                                                   const unsigned int num_width = 9,
                                                   const bool output_errors = true)
  {
    if (!create_or_check_folder(folder, output_errors))
      {
        return ErrorState::WriteError;
      }

    std::ostringstream event_ID_format;
    event_ID_format << std::setfill('0') << std::setw(num_width) << event_number;
    const std::string event_ID = event_ID_format.str();

    auto filename = [&] (const std::string & ext)
    {
      return folder / build_filename(prefix, event_ID, suffix, ext);
    };

    if (EventInformation::write_cluster_info(filename("clusters"), clusters) != ErrorState::OK)
      {
        return ErrorState::WriteError;
      }
    return ErrorState::OK;
  }

  struct FolderLoad
  {
    std::map<std::string, CaloRecGPU::Helpers::CPU_object<CaloRecGPU::GeometryArr>> geometry;
    std::map<std::string, CaloRecGPU::Helpers::CPU_object<CaloRecGPU::CellNoiseArr>> noise;
    std::map<std::string, CaloRecGPU::Helpers::CPU_object<CaloRecGPU::ClusterInfoArr>> cluster_info;
    std::map<std::string, CaloRecGPU::Helpers::CPU_object<CaloRecGPU::CellInfoArr>> cell_info;
    std::map<std::string, CaloRecGPU::Helpers::CPU_object<CaloRecGPU::CellStateArr>> cell_state;
  };

  /*! @class FolderLoadOptions

      The members of this structure should all be initialised by default to @c false.
      It is the function caller's responsibility to specify what should be loaded.
  */
  struct FolderLoadOptions
  {
    bool load_cluster_info = false,
         load_cell_state = false,
         load_cell_info = false,
         load_geometry = false,
         load_noise = false;

    static constexpr FolderLoadOptions None()
    {
      FolderLoadOptions ret;
      return ret;
      //By design, all options are initialized as false...
    }
    static constexpr FolderLoadOptions All()
    {
      FolderLoadOptions ret{true, true, true, true, true};
      return ret;
    }
  };


  /*! @param filter_function receives a std::string (the filename),
             returns `true` if the file should be filtered out.
  */
  template <class F>
  inline static FolderLoad load_folder_filter(F && filter_function,
                                              //Receives a std::string (the filename),
                                              //returns `true` if the file should be filtered out.
                                              const std::filesystem::path & folder,
                                              int max_events = -1,
                                              const FolderLoadOptions & flo = FolderLoadOptions::None(),
                                              const bool output_messages = true)
  {
    FolderLoad ret;
    if (!std::filesystem::is_directory(folder))
      {
        if (output_messages)
          {
            std::cout << "ERROR: '" << folder << "' is not a valid folder." << std::endl;
          }
        return ret;
      }
    std::set<std::string> read_one_part_of_v1_cells;
    int event_count = 0;
    for (const std::filesystem::path & file : std::filesystem::directory_iterator(folder))
      {
        if ( max_events > 0 && event_count >= max_events &&
             ret.geometry.size() > 0 && ret.noise.size() > 0 )
          {
            break;
          }

        const bool can_load_events = (max_events < 0) || (event_count < max_events);
        const std::string filename = file.stem().native();

        if (filter_function(filename))
          {
            continue;
          }

        auto check_error = [&](const ErrorState & es, const std::string & str)
        {
          if (es == ErrorState::OK)
            {
              return false;
            }
          if (output_messages)
            {
              std::cout << "ERROR: '" << file << "' is not a valid " << str << " file (" << (int) es << ")." << std::endl;
            }
          return true;
        };

        auto output_loading_message = [&](const std::string & str)
        {
          if (output_messages)
            {
              std::cout << "Loaded " << str << " from '" << file << "'." << std::endl;
            }
        };

        if (file.extension() == ".geometry")
          {
            if (!flo.load_geometry)
              {
                continue;
              }
            CaloRecGPU::Helpers::CPU_object<CaloRecGPU::GeometryArr> tempgeo(true);
            if (check_error(ConstantInformation::read_geometry(file, tempgeo), "geometry"))
              {
                continue;
              }
            ret.geometry[filename] = std::move(tempgeo);
            output_loading_message("geometry");
          }
        else if (file.extension() == ".cellstate")
          {
            if (!flo.load_cell_state || !can_load_events)
              {
                continue;
              }
            CaloRecGPU::Helpers::CPU_object<CaloRecGPU::CellStateArr> tempcellstate(true);
            if (check_error(EventInformation::read_cell_state(file, tempcellstate), "cell state"))
              {
                continue;
              }
            ret.cell_state[filename] = std::move(tempcellstate);
            output_loading_message("cell state");
            if ((ret.cluster_info.count(filename) > 0 || !flo.load_cluster_info) &&
                (ret.cell_info.count(filename) > 0 || !flo.load_cell_info))
              {
                ++event_count;
              }
          }
        else if (file.extension() == ".cellinfo")
          {
            if (!flo.load_cell_info || !can_load_events)
              {
                continue;
              }
            CaloRecGPU::Helpers::CPU_object<CaloRecGPU::CellInfoArr> tempcellinfo(true);
            if (check_error(EventInformation::read_cell_info(file, tempcellinfo), "cell info"))
              {
                continue;
              }
            ret.cell_info[filename] = std::move(tempcellinfo);
            output_loading_message("cell info");
            if ((ret.cluster_info.count(filename) > 0 || !flo.load_cluster_info) &&
                (ret.cell_state.count(filename) > 0 || !flo.load_cell_state))
              {
                ++event_count;
              }
          }
        else if (file.extension() == ".clusterinfo")
          {
            if (!flo.load_cluster_info || !can_load_events)
              {
                continue;
              }
            CaloRecGPU::Helpers::CPU_object<CaloRecGPU::ClusterInfoArr> tempclu(true);
            if (check_error(EventInformation::read_cluster_info(file, tempclu), "cluster info"))
              {
                continue;
              }
            ret.cluster_info[filename] = std::move(tempclu);
            output_loading_message("cluster info");
            if ((ret.cell_state.count(filename) > 0 || !flo.load_cell_state) &&
                (ret.cell_info.count(filename) > 0 || !flo.load_cell_info))
              {
                ++event_count;
              }
          }
        else if (file.extension() == ".noise")
          {
            if (!flo.load_noise)
              {
                continue;
              }
            CaloRecGPU::Helpers::CPU_object<CaloRecGPU::CellNoiseArr> tempnois(true);
            if (check_error(ConstantInformation::read_noise(file, tempnois), "noise"))
              {
                continue;
              }
            ret.noise[filename] = std::move(tempnois);
            output_loading_message("noise");
          }
      }
    return ret;
  }

  inline static FolderLoad load_folder(const std::filesystem::path & folder,
                                       int max_events = -1,
                                       const FolderLoadOptions & flo = FolderLoadOptions::None(),
                                       const bool output_messages = true)
  {
    return load_folder_filter([&](const std::string &)
    {
      return false;
    },
    folder, max_events, flo, output_messages);
  }

};


#endif //CALORECGPU_STANDALONEDATAIO_H
