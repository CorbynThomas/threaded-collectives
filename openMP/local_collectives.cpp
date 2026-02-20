/******************************************************************************
* FILE: local_collectives.cpp
* DESCRIPTION:
*   functions to scatter and gather over threads on openMP
* AUTHOR: Corbyn Thomas
* LAST REVISED: 01/21/2026
******************************************************************************/
using namespace std;
#include <iostream>
#include <functional>
#include <cstdlib> 
#include <cmath>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define MASTER 0

void local_scatter(const int* send_data, int send_count, int* rec_data, int root);

void local_gather(const int* send_data, int send_count, int* rec_data, int root);

int main(int argc, char** argv)
{
    

    #pragma omp parallel num_threads(3)
    {
        int tid = omp_get_thread_num();
        int numt = omp_get_num_threads();

        int * send = nullptr;
        int * recv = new int[1];

        if(tid == 0)
        {
            int * send = new int[3];
            for(int i = 0; i < 3; i++)
            {
                send[i] = i;
            }
        }
        local_scatter(send, 1, recv, 0);

        for(int i = 0; i < numt; i++)
        {
            #pragma omp barrier
            if(tid == i)
            {
                cout << "This is thread #" << tid << " my number is " << recv[0] << endl;
            }
        }

        delete[] send;
        delete[] recv;
    }

    return 0;
}

//to be used within a parrallelized environment
void local_scatter(const int* send_data, int send_count, int* rec_data, int root)
{
    int tid  = omp_get_thread_num();
    int numt = omp_get_num_threads();

    // Shared temporary buffer
    static int* shared = new int[numt * send_count];

    if(tid == root)
    {
        for (int t = 0; t < numt; ++t)
        {
            memcpy(shared + (t * send_count * sizeof(int)), &send_data[t * send_count * sizeof(int)], send_count * sizeof(int));
        }
    }

    #pragma omp barrier

    // Each thread copies its chunk
    memcpy(rec_data, &shared[tid * send_count * sizeof(int)], send_count * sizeof(int));

    #pragma omp barrier
    #pragma omp single
    delete[] shared;
}
