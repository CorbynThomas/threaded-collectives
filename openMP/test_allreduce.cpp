/******************************************************************************
* FILE: test_allreduce.cpp
* DESCRIPTION:
*   unit tests for both thread aware reduce
* AUTHOR: Corbyn Thomas
* LAST REVISED: 2/6/2025
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

    constexpr int sendcount = 96;

    int num_threads = 0;

    #pragma omp parallel num_threads(12)
    {
        #pragma omp single
            num_threads = omp_get_num_threads();
        int tid = omp_get_thread_num();

        std::vector<int> buf;
        buf.resize(sendcount);
        for(int i = 0; i < sendcount; i++)
        {
            buf[i] = tid + i;
        }

        // Each thread sends a unique value
        std::vector<int> recbuf;
        recbuf.resize(sendcount, -12);
        
        //output each threads data
        for (int r = 0; r < size; ++r)
        {
            for (int t = 0; t < num_threads; ++t)
            {
                best_barrier();

                if(rank == r && tid == t)
                    std::cout << "Rank " << r << ", Thread " << t << " value: " << buf[0] << " " << buf[1] << " " << buf[2] << std::endl;
            }
            best_barrier();
                
            if(rank == 0 && tid == 0)
            {
                std::cout << "\n";
            }

            best_barrier();
        }

        best_barrier();

        double time1, time2, time3;
        time1 = MPI_Wtime(); //getting the first time

        best_barrier();

        int rc = thread_aware_allreduce(thread_allreduce, buf.data(), recbuf.data(), sendcount, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);
        assert(rc == MPI_SUCCESS);

        best_barrier();

        time2 = MPI_Wtime();
        time2 = time2 - time1;

        if(tid == 0)
            MPI_Reduce(&time2, &time3, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD); //reduce the time from each worker to master

        best_barrier();

        if(rank == MASTER && tid == 0)
        {
            cout << "\nThe time the program took to run was " << time3/size << " seconds with " << sendcount << " sends and " << num_threads << " threads" << endl;
        }

        if(rank == 0 && tid == 0)
        {
            std::cout << endl;
        }

        
        for (int r = 0; r < size; ++r)
        {
            for (int t = 0; t < num_threads; ++t)
            {
                best_barrier();

                if(rank == r && tid == t)
                    std::cout << "Rank " << r << ", Thread " << t << " value: " << recbuf[0] << " " << recbuf[1] << " " << recbuf[2] << std::endl;
            }
            best_barrier();
                
            if(rank == 0 && tid == 0)
            {
                std::cout << "\n";
            }

            best_barrier();
        }
        
    }

    MPI_Finalize();
    return 0;
}
