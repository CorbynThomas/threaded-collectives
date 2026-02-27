/******************************************************************************
* FILE: test_bcast.cpp
* DESCRIPTION:
*   unit tests for both thread aware broadcast
* AUTHOR: Corbyn Thomas
* LAST REVISED: 2/3/2025
******************************************************************************/
#include <mpi.h>
#include <omp.h>
#include <cassert>
#include <vector>
#include "thread_functions.cpp" 

#define MASTER 0

int main(int argc, char** argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if(provided < MPI_THREAD_MULTIPLE)
    {
        printf("The threading support level is lesser than that demanded.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    constexpr int sendcount = 1;
    constexpr int root = MASTER;

    int num_threads = 0;

    #pragma omp parallel num_threads(3)
    {
        num_threads = omp_get_num_threads();
        int tid = omp_get_thread_num();

        std::vector<int> buf;
        buf.resize(sendcount, -1);
        fill(buf.begin(), buf.end(), 12);

        // Each thread sends a unique value
        std::vector<int> sendbuf;
        if(rank == MASTER)
        {
            fill(buf.begin(), buf.end(), tid * 10);
        }
        
        
        if(rank == 0)
        {
            // Already inside #pragma omp parallel
            for (int t = 0; t < num_threads; ++t)
            {
                #pragma omp barrier
                if (tid == t)
                {
                    std::cout << "Rank " << rank << ", Thread " << tid << " values: " << buf[0] << std::endl;
                }
            }
        }

        best_barrier();

        int rc = thread_aware_broadcast(thread_broadcast, buf.data(), sendcount, MPI_INT, root, MPI_COMM_WORLD);
        assert(rc == MPI_SUCCESS);

        if(rank == 0 && tid == 0)
        {
            std::cout << "\n";
        }

        for (int r = 0; r < size; ++r)
        {
            for (int t = 0; t < num_threads; ++t)
            {
                #pragma omp barrier
                #pragma omp master
                    MPI_Barrier(MPI_COMM_WORLD);
                if(rank == r && tid == t)
                    std::cout << "Rank " << r << ", Thread " << t << " value: " << buf[0] << ::endl;
            }
            #pragma omp barrier
            #pragma omp master
                MPI_Barrier(MPI_COMM_WORLD);
            #pragma omp barrier
                
            if(rank == 0 && tid == 0)
            {
                std::cout << "\n";
            }

            #pragma omp barrier
            #pragma omp master
                MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    /*
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Validate results on root
    if (rank == root)
    {
        #pragma omp master
        {
            for (int r = 0; r < size; ++r)
            {
                for (int t = 0; t < num_threads; ++t)
                {
                    int idx = (r * num_threads + t) * sendcount;
                    int expected = r * 1000 + t;
                    int actual = recvbuf[idx];
                
                    if (actual != expected)
                    {
                        std::cerr << "Mismatch at rank=" << r << " tid=" << t << " expected=" << expected << " got=" << actual << std::endl;
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                }
            }
        }

        std::cout << "[PASS] gpu_aware_gather(thread_gather) validated successfully\n";
    }
    */

    MPI_Finalize();
    return 0;
}
