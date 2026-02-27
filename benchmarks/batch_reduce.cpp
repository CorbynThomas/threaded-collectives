/******************************************************************************
* FILE: test_batchReduce.cpp
* DESCRIPTION:
*   unit tests for batch reduce
* AUTHOR: Corbyn Thomas
* LAST REVISED: 2/17/2025
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
    else if(argc != 2)
    {
        std::cout << "\nIncorrect amount of command line arguments";
        return -1;
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int sendcount = std::stoi(argv[1]);

    int num_threads = 0;

    #pragma omp parallel
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
        if(rank == 0)
        {
            recbuf.resize(sendcount, -12);
        }

        //warmup run
        best_barrier();
        int rc = batch_reduce(buf.data(), recbuf.data(), sendcount, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);
        assert(rc == MPI_SUCCESS);
        best_barrier();
        

        double time1, time2, time3;
        double timeTotal = 0;
        for(int i = 0; i < 75; i++)
        {
            time1 = MPI_Wtime(); //getting the first time
            best_barrier();

            int rc = batch_reduce(buf.data(), recbuf.data(), sendcount, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);
            assert(rc == MPI_SUCCESS);

            time2 = MPI_Wtime();
            time2 = time2 - time1;
            best_barrier();

            if(tid == 0)
            {
                MPI_Reduce(&time2, &time3, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD); //reduce the time from each worker to master
                timeTotal += time3;
            }

            best_barrier();
        }

        if(rank == MASTER && tid == 0)
        {
            cout << "\nThe time the program took to run on average over 75 runs was " << timeTotal/(size * 75) << " seconds with " << sendcount << " sends, " << num_threads << " threads and " << size << " processes" << endl;
        }
    }

    MPI_Finalize();
    return 0;
}
