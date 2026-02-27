/******************************************************************************
* FILE: alltest.cpp
* DESCRIPTION:
*   unit tests for gpu aware alltoall function
* AUTHOR: Corbyn Thomas
* LAST REVISED: 01/21/2026
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

        std::vector<int> recvbuf;
        recvbuf.resize(size, -1);
        fill(recvbuf.begin(), recvbuf.end(), 12);

        int tid = omp_get_thread_num();

        // Each thread sends a unique value
        std::vector<int> sendbuf;
        sendbuf.resize(size, -1);
        for(int i = 0; i < size; i++)
        {
            sendbuf[i] = rank * 18 + tid * 6 + i;
        }
        
        for (int r = 0; r < size; ++r)
        {
            #pragma omp master
                MPI_Barrier(MPI_COMM_WORLD);
        
            if (rank == r)
            {
                // Already inside #pragma omp parallel
                for (int t = 0; t < num_threads; ++t)
                {
                    #pragma omp barrier
                    if (tid == t)
                    {
                        std::cout << "Rank " << rank << ", Thread " << tid << " value: ";
                        for(int value = 0; value < size; value++)
                        {
                            cout << sendbuf[value] << " ";
                        }
                        cout << endl;
                    }
                }
            }
        }
        #pragma omp master
            MPI_Barrier(MPI_COMM_WORLD);

        int rc = thread_aware_alltoall(thread_alltoall, sendbuf.data(), sendcount, MPI_INT, recvbuf.data(), sendcount, MPI_INT, MPI_COMM_WORLD);
        assert(rc == MPI_SUCCESS);

        for (int r = 0; r < size; ++r)
        {
            #pragma omp master
                MPI_Barrier(MPI_COMM_WORLD);
        
            if (rank == r)
            {
                // Already inside #pragma omp parallel
                for (int t = 0; t < num_threads; ++t)
                {
                    #pragma omp barrier
                    if (tid == t)
                    {
                        std::cout << "Rank " << rank << ", Thread " << tid << " value: ";
                        for(int value = 0; value < size; value++)
                        {
                            cout << recvbuf[value] << " ";
                        }
                        cout << endl;
                    }
                }
            }
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
