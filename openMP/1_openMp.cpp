#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>

using namespace std;
 
/**
 * @brief Illustrates how to initialise the MPI environment with multithreading
 * support.
 * @details This application initialised MPI and asks for the 
 * MPI_THREAD_SERIALIZED thread support level. It then compares it with the
 * thread support level provided by the MPI implementation.
 * Start using scatter/gather in mpi. Make a generic interface/function
 **/
int main(int argc, char* argv[])
{
    // Initilialise MPI and ask for thread support
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

    cout << "Rank " << rank << " staring OpenMP with thread multiple\n";

    #pragma omp parallel num_threads(2)
    {
        int tid = omp_get_thread_num();

        if(tid == 0)
        {
            int value = rank + 100;
            cout << "Rank " << rank << " Thread " << " : send value " << value << endl;
            MPI_Send(&value, 1, MPI_INT, (rank + 1) % size, 0, MPI_COMM_WORLD);
        }
        else
        {
            int recv;
            MPI_Recv(&recv, 1, MPI_INT, (rank - 1 + size) % size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            cout << "Rank " << rank << " Thread " << " : recieved value " << recv << endl;        
        }
    }

    // Tell MPI to shut down.
    MPI_Finalize();
 
    return EXIT_SUCCESS;
}