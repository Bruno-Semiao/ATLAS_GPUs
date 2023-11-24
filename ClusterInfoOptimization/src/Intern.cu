#include "BasicGPUClusterInfoCalculatorImpl.cu"
#include "StandaloneDataIO.h"
#include "CUDA_Helpers.cu"
#include "DataHolders.cu" 
#include "IGPUKernelSizeOptimizer.h"

#define N_EVENTS 999999

using namespace std;

//--Include-- 
//Folder load
//file name at BasicGPUClusterInfoCalculatorImpl0500.cu

//~55 minutes to run 500 events * 256 combinations

class Optimizer: public IGPUKernelSizeOptimizer{
     void register_kernels(const std::string & tool_name,
                                const int number,
                                void ** kernels,
                                const int * blocksize_hints,
                                const int * gridsize_hints,
                                const int * max_total_threads,
                                const int offset = 0)
    {
    //Pode ser uma no-op, porque já sabes os kernels que queres lançar.
    }

    /** @brief Retrieve the (hopefully optimal) kernel launch configuration.*/
    CUDAKernelLaunchConfiguration get_launch_configuration(const std::string & name,
                                                                    const int number = 0,
                                                                    const int dynamic_memory = 0) const
    {
        if (name == "BasicClusterInfoCalculator") {
            switch (number)
            {
                case 0:
                    KernelConfig.grid_x  = gridSize;
                    KernelConfig.grid_y  = 1;
                    KernelConfig.grid_z  = 1;
                    KernelConfig.block_x = blockSize;
                    KernelConfig.block_y = 4;
                    KernelConfig.block_z = 1;
                case 1:
                    KernelConfig.grid_x  = 1;
                    KernelConfig.grid_y  = 1;
                    KernelConfig.grid_z  = 1;
                    KernelConfig.block_x = 1;
                    KernelConfig.block_y = 1;
                    KernelConfig.block_z = 1;
                case 2:
                    KernelConfig.grid_x  = gridSize;
                    KernelConfig.grid_y  = 1;
                    KernelConfig.grid_z  = 1;
                    KernelConfig.block_x = blockSize;
                    KernelConfig.block_y = 1;
                    KernelConfig.block_z = 1;
                default:
                    break;
            }
            return KernelConfig;   
        }

        /*if (name == "BasicClusterInfoCalculator") {
            switch (number)
            {
            case 0:
                KernelConfig.grid_x = 0;
            case 1:
                return 1
            default:
                break;
            }
        }*/
        return {};
    }

    bool can_use_cooperative_groups() const
    {
        return false;
    }

    bool can_use_dynamic_parallelism() const
    {
        return true;
    }

    bool use_minimal_kernel_sizes() const
    {
        return false;
    }
};


int main()
{
    //Setup Simulation Data
    auto folder = StandaloneDataIO::load_folder("/users3/ship/u22brunosemiao/ClusterInfoOptimization/simulations", -1, StandaloneDataIO::FolderLoadOptions::All(), true);//StandaloneDataIO::load_folder("/user/n/nunosfernandes/Data", -1, StandaloneDataIO::FolderLoadOptions::All(), true);

    CaloRecGPU::EventDataHolder    eHolder;
    CaloRecGPU::ConstantDataHolder cHolder;

    eHolder.allocate();

    cHolder.m_geometry_dev   = folder.geometry.begin()->second;

    cHolder.m_cell_noise_dev = folder.noise.begin()->second;

    auto it_info             = folder.cell_info.begin();
    auto it_cluster          = folder.cluster_info.begin();
    auto it_state            = folder.cell_state.begin();
   
    int it_loop = 0;

    //Setup Optimizer
    Optimizer optimizer;

    for(int dG = 32; dG <= 256; dG+=32)
    {
        for(int dB = 32; dB <= 256; dB+=32)
        {   
            it_info     = folder.cell_info.begin();
            it_cluster  = folder.cluster_info.begin();
            it_state    = folder.cell_state.begin();
            it_loop     = 0;
            optimizer.set_kernel_sizes(dG, dB);
            //optimizer.get_launch_configuration("BasicClusterInfoCalculator", 0, 0, kernelSizes);
            while(it_info!=folder.cell_info.end() && it_state!=folder.cell_state.end() && it_cluster!=folder.cluster_info.end())
            {
                eHolder.m_cell_info_dev   = it_info->second;
                eHolder.m_cell_state_dev  = it_state->second;
                eHolder.m_clusters_dev    = it_cluster->second;
                
                calculateClusterProperties(eHolder, (ClusterInfoCalculatorTemporaries*) ((void*) (ClusterMomentsArr*) eHolder.m_moments_dev), cHolder, optimizer, false, true, -1, {});
                cudaDeviceSynchronize();
                
                it_info++;
                it_state++;
                it_cluster++;
                it_loop++;
                if(it_loop>=N_EVENTS){break;}
            }
            //cout << "Grid: " << dG << "Block: " << dB << endl;
        }
    }
    
    
}