/******************************************************************************
* FILE: monte_thread.cpp
* DESCRIPTION:
*   MPI and Openmp code to approxomate the value of pi using the Monte Carlo method using threading with command line interaction
* AUTHOR: Corbyn Thomas
* LAST REVISED: 12/1/2025
******************************************************************************/
using namespace std;
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <mpi.h>
#include "thread_functions.cpp" 
#define  MASTER		0

int main(int argc, char *argv[]) {

    int dest, source, tag = 1;
    long long local; //used to store recieved data

    double time1, time2, time3;

    long long npoints = atoll(argv[1]); // Number of random points to generate
    int num_tasks; //number of tasks
    int taskid; //master or worker
    int max_threads = omp_get_max_threads(); //max number of threads per rank
    int provided;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if(provided < MPI_THREAD_MULTIPLE)
    {
        printf("The threading support level is lesser than that demanded.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
    MPI_Comm_rank(MPI_COMM_WORLD,&taskid);

    if(taskid == MASTER)
    {
        if (argc > 1)
        {
            printf("Program name: %s\n", argv[0]);
            printf("Number of points: %s\n", argv[1]);
        } 
        else
        {
            printf("Not enough command-line arguments provided.\n");
            return 1;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    long long points_per_rank = npoints/num_tasks;
    long long points_per_thread = points_per_rank/max_threads;

    
    time1 = MPI_Wtime(); //getting the first time
    #pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        long long circle_count = 0; // number of points in the circle unique to each thread
        //each thread tests points_per_thread number of points
        
        srand(time(0) + 1000 * (tid + taskid));

        for (long long i = 0; i < points_per_thread; i++)
        {
            // Generate random x and y coordinates between 0 and 1
            double x = static_cast<double>(rand()) / RAND_MAX;
            double y = static_cast<double>(rand()) / RAND_MAX;
            // Calculate the distance from the origin (0,0)
            double distance_squared = (x * x) + (y * y);
            // Check if the point falls inside the unit circle (radius 1)
            if (distance_squared <= 1.0) {
                circle_count++;
            }
        }

        if(taskid == 0 && tid == 0)
            cout << "Number of points per task: " << points_per_rank << " Points per thread: " << points_per_thread << endl;
        
        long long* sendBuffer = new long long[1];
        sendBuffer[0] = circle_count;
        long long* recBuffer = nullptr;
        //each rank's master thread has a recieve buffer sized to hold the circle count from each thread
        #pragma omp master
        {
            recBuffer = new long long[omp_get_num_threads()];
        }
        
        #pragma omp barrier
        #pragma omp master
            MPI_Barrier(MPI_COMM_WORLD);
        #pragma omp barrier

        //each rank gather to its own main thread then add each rank together.
        for(int i = 0; i < num_tasks; i++)
        {
            thread_gather(sendBuffer, 1, MPI_LONG_LONG, recBuffer, 1, MPI_LONG_LONG, i, i, MPI_COMM_WORLD);
        }

        for(int i = 0; i < num_tasks; i++)
        {
            #pragma omp master
                MPI_Barrier(MPI_COMM_WORLD);
            if(tid == 0 && taskid == i)
            {
                cout << "Task #" << i << " Thread #0 My recieve buffer: ";
                for(int j = 0; j < omp_get_num_threads(); j++)
                {
                    cout << recBuffer[j] << " ";
                }
                cout << endl;
            }
        }

        #pragma omp master
        {
            circle_count = 0; 
            for(int i = 0; i < omp_get_num_threads(); i++)
            {
                circle_count += recBuffer[i];
            }
            local = circle_count;
        }

        
        for(int i = 0; i < num_tasks; i++)
        {
            #pragma omp master
                MPI_Barrier(MPI_COMM_WORLD);
            if(tid == 0 && taskid == i)
            {
                cout << "Task #" << i << " Thread #0 My added circle count: " << circle_count << endl;
            }
        }
    }

    long long global_count = 0;
    MPI_Reduce(&local, &global_count, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (taskid == 0) { // rank 0
        double pi_estimate = 4.0 * global_count / npoints;
        cout << "Pi estimate = " << pi_estimate << endl;
    }
    
    MPI_Finalize();

    return 0;
}