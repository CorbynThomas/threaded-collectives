/******************************************************************************
* FILE: thread_functions.cpp
* DESCRIPTION:
*   create working and efficient gather and scatter functions to work on threads similar to how MPI_gather and MPI_scatter works between ranks
* AUTHOR: Corbyn Thomas
* LAST REVISED: 11/20/2025
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
#include <mpi.h>
#define  MASTER		0

using alltoall_ftn = std::function<int(const void*, const int, MPI_Datatype, void*, const int, MPI_Datatype, MPI_Comm)>;
using allreduce_ftn = std::function<int(const void*, void* , int, MPI_Datatype, MPI_Op, int, MPI_Comm)>;
using broadcast_ftn = std::function<int(void*, const int, MPI_Datatype, int, MPI_Comm)>;
using gather_ftn = std::function<int(const void*, const int, MPI_Datatype, void*, const int, MPI_Datatype, int, MPI_Comm)>;
using scatter_ftn = std::function<int(const void*, const int, MPI_Datatype, void*, const int, MPI_Datatype, int, MPI_Comm)>;
using reduce_ftn = std::function<int(const void*, void* , int, MPI_Datatype, MPI_Op, int, MPI_Comm)>;

int thread_aware_alltoall(alltoall_ftn f, const void* sendbuf, const int sendcount, MPI_Datatype sendtype, void* recvbuf, const int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
{
    int ans = f(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
    return ans;
}

int thread_aware_allreduce(allreduce_ftn f, const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)
{
    int ans = f(sendbuf, recvbuf, count, datatype, op, root, comm);
    return ans;
}

int thread_aware_broadcast(broadcast_ftn f, void* buffer, const int count, MPI_Datatype datatype, int root, MPI_Comm comm)
{
    int ans = f(buffer, count, datatype, root, comm);
    return ans;
}

int thread_aware_gather(gather_ftn f, const void* sendbuf, const int sendcount, MPI_Datatype sendtype, void* recvbuf, const int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
{
    int ans = f(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
    return ans;
}

int thread_aware_scatter(scatter_ftn f, const void* sendbuf, const int sendcount, MPI_Datatype sendtype, void* recvbuf, const int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
{
    int ans = f(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
    return ans;
}

int thread_aware_reduce(reduce_ftn f, const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)
{
    int ans = f(sendbuf, recvbuf, count, datatype, op, root, comm);
    return ans;
}

void best_barrier()
{
    #pragma omp barrier
    #pragma omp master
        MPI_Barrier(MPI_COMM_WORLD);
}

//these functions need to be used while a #pragma omp parallel is in affect
int thread_allreduce(const void* send_data, void* rec_data, const int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm communicator)
{
    int task;
    MPI_Comm_rank(communicator, &task);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int type_size;
    MPI_Type_size(datatype, &type_size);

    //for every thread on every rank send to the root.

    
    const char* send_ptr = static_cast<const char*>(send_data);
    int tid = omp_get_thread_num();

    if(task != root)
    {
        #pragma omp barrier
        MPI_Send(send_ptr, count, datatype, root, tid, communicator);
    }

    
    #pragma omp barrier

    //Each thread on root recieves a collection of the send data from the corresponding threads in other ranks  
    if(task == root)
    {
        char* stored_ptr = (char*)malloc((size_t)size * type_size * count); //malloc a storage for each thread on root to hold all the gathered data to be looped over and summed, or maxed.

        for(int rank = 0; rank < size; rank++)
        {
            char* offset_ptr = stored_ptr + (rank * count * type_size);

            if(rank == root)
            {
                memcpy(offset_ptr, send_ptr, count * type_size);
            }
            else
            {
                MPI_Recv(offset_ptr, count, datatype, rank, tid, communicator, MPI_STATUS_IGNORE);
            }
        }


        //cast a way to store the data depending on the MPI_Datatype
        if(datatype == MPI_INT)
        {
            int* base = (int*)stored_ptr;
            if(op == MPI_SUM)
            {
                for(int j = 0; j < count; j++)
                {
                    int sum = 0;
                    for (int i = 0; i < size; i++)
                    {
                        #pragma omp barrier
                        sum += base[j + i * count];
                    }
                
                    static_cast<int*>(rec_data)[j] = sum;
                }
            }
        }

        //root sends to each other rank's threads what it has gathered
        for(int i = 0; i < size; i++)
        {
            #pragma omp barrier
            MPI_Send(rec_data, count, datatype, i, tid, communicator);
        }

    }

    best_barrier();

    if(task != root)
    {
        MPI_Recv(rec_data, count, datatype, root, tid, communicator, MPI_STATUS_IGNORE);
    }

    return MPI_SUCCESS;
}

int thread_broadcast(void* buffer, const int count, MPI_Datatype datatype, int root, MPI_Comm communicator)
{
    int task;
    MPI_Comm_rank(communicator, &task);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int type_size;
    MPI_Type_size(datatype, &type_size);

    //distribute the piece of data on each of the root's threads to each other thread evenly. ex: 0.1 = 1; 0.1 = 1 1.1 = 1 2.1 = 1 3.1 = 1

    
    int num_threads = omp_get_num_threads();
    int tid = omp_get_thread_num();
    char* ptr = static_cast<char*>(buffer);

    if(task == root)
    {
        for(int i = 0; i < size; i++)
        {
            if(task != i)
            {
                MPI_Send(ptr, count, datatype, i, tid, communicator);
            }
        }
    }

    best_barrier();

    if(task != root)
    {
        MPI_Recv(ptr, count, datatype, root, tid, communicator, MPI_STATUS_IGNORE);
    }

    return MPI_SUCCESS;
}

int thread_gather(const void* send_data, const int send_count, MPI_Datatype send_datatype, void* rec_data, const int rec_count, MPI_Datatype rec_datatype, int root, MPI_Comm communicator)
{
    int task;
    MPI_Comm_rank(communicator, &task);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int type_size;
    MPI_Type_size(send_datatype, &type_size);

    //for every thread on every rank send to the root.

    
    const char* send_ptr = static_cast<const char*>(send_data);
    int tid = omp_get_thread_num();
    if(task != root)
    {
        for(int j = 0; j < omp_get_num_threads(); j++)
        {
            #pragma omp barrier
            if(tid == j)
            {
                MPI_Send(send_ptr, send_count, send_datatype, root, tid, MPI_COMM_WORLD);
            }
        }
    }

    
    #pragma omp barrier

    //The master thread on the root recieves all of them    
    if(task != root)
    {
        return MPI_SUCCESS;
    }

    char* rec_ptr = static_cast<char*>(rec_data);
    int num_threads = omp_get_num_threads();
    for(int rank = 0; rank < size; rank++)
    {
        for(int tid = 0; tid < num_threads; tid++)
        {
            char* offset_ptr = rec_ptr + ((rank * num_threads + tid) * send_count * type_size);
            if(rank == root && tid == omp_get_thread_num())
            {
                memcpy(offset_ptr, send_ptr, send_count * type_size);
            }
            else
            {
                #pragma omp master
                {
                    MPI_Recv(offset_ptr, rec_count, rec_datatype, rank, tid, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        }
    }

    return MPI_SUCCESS;
}

int thread_scatter(const void* send_data, const int send_count, MPI_Datatype send_datatype, void* rec_data, const int rec_count, MPI_Datatype rec_datatype, int root, MPI_Comm communicator)
{
    int task;
    MPI_Comm_rank(communicator, &task);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int type_size;
    MPI_Type_size(send_datatype, &type_size);

    //distribute the data of each of the root's threads to each other thread evenly. ex: 0.1 = 1 2 3 4; 0.1 = 1 1.1 = 2 2.1 = 3 3.1 = 4

    
    char* rec_ptr = static_cast<char*>(rec_data);
    int num_threads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    if(task == root)
    {
        const char* send_ptr = static_cast<const char*>(send_data);

        for(int i = 0; i < size; i++)
        {
            char* offset_ptr = rec_ptr + (i * send_count * type_size);
            if(task == i)
            {
                memcpy(offset_ptr, &send_ptr[i * send_count * type_size], send_count * type_size);
            }
            else
            {
                MPI_Send(&send_ptr[i * send_count * type_size], send_count, send_datatype, i, tid, communicator);
            }
        }
    }

    #pragma omp barrier
    #pragma omp master
        MPI_Barrier(communicator);

    if(task != root)
    {
        MPI_Recv(rec_ptr, rec_count, rec_datatype, root, tid, communicator, MPI_STATUS_IGNORE);
    }

    return MPI_SUCCESS;
}

int thread_alltoall(const void* send_data, const int send_count, MPI_Datatype send_datatype, void* rec_data, const int rec_count, MPI_Datatype rec_datatype, MPI_Comm communicator)
{
    int task;
    MPI_Comm_rank(communicator, &task);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int type_size;
    MPI_Type_size(send_datatype, &type_size);

    //all to all, but using threads instead. rank 0's thread #1 will consist of 0.1 1.1 2.1 3.1 ...
    const char* send_ptr = static_cast<const char*>(send_data);
    char* rec_ptr = static_cast<char*>(rec_data);
    int num_threads = omp_get_num_threads();
    int tid = omp_get_thread_num();
    

    //each thread loops through each rank and sends it's data to the corresponding thread in that rank
    for(int i = 0; i < size; i++)
    {
        if(i != task) 
        {
            MPI_Send(&send_ptr[i * send_count * type_size], send_count, send_datatype, i, tid, communicator);
        } 
    }

    
    #pragma omp barrier
    #pragma omp master
        MPI_Barrier(communicator);

    //Each thread now recieves what it gets from each other rank's thread or copies from itself to the correct place
    for(int rank = 0; rank < size; rank++)
    {
        char* offset_ptr = rec_ptr + (rank * send_count * type_size);
        if(rank == task) 
        {
            memcpy(offset_ptr, &send_ptr[rank * send_count * type_size], send_count * type_size);
        }
        else
        {
            
            MPI_Recv(offset_ptr, rec_count, rec_datatype, rank, tid, communicator, MPI_STATUS_IGNORE);
        }

    }
    
    return MPI_SUCCESS;
}

int thread_reduce(const void* send_data, void* rec_data, const int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm communicator)
{
    int task;
    MPI_Comm_rank(communicator, &task);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int type_size;
    MPI_Type_size(datatype, &type_size);

    //for every thread on every rank send to the root.

    
    const char* send_ptr = static_cast<const char*>(send_data);
    int tid = omp_get_thread_num();

    if(task != root)
    {
        #pragma omp barrier
        MPI_Send(send_ptr, count, datatype, root, tid, communicator);
    }

    
    #pragma omp barrier

    //Each thread on root recieves a collection of the send data from the corresponding threads in other ranks  
    if(task != root)
    {
        return MPI_SUCCESS;
    }

    char* stored_ptr = (char*)malloc((size_t)size * type_size * count); //malloc a storage for each thread on root to hold all the gathered data to be looped over and summed, or maxed.

    for(int rank = 0; rank < size; rank++)
    {
        char* offset_ptr = stored_ptr + (rank * count * type_size);

        if(rank == root)
        {
            memcpy(offset_ptr, send_ptr, count * type_size);
        }
        else
        {
            MPI_Recv(offset_ptr, count, datatype, rank, tid, communicator, MPI_STATUS_IGNORE);
        }
    }


    //cast a way to store the data depending on the MPI_Datatype
    if(datatype == MPI_INT)
    {
        int* base = (int*)stored_ptr;
        if(op == MPI_SUM)
        {
            for(int j = 0; j < count; j++)
            {
                int sum = 0;
                for (int i = 0; i < size; i++)
                {
                    #pragma omp barrier
                    sum += base[j + i * count];
                }
            
                static_cast<int*>(rec_data)[j] = sum;
            }
        }
    }

    return MPI_SUCCESS;
}

int batch_reduce(const void* send_data, void* rec_data, const int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm communicator)
{
    int task;
    MPI_Comm_rank(communicator, &task);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int type_size;
    MPI_Type_size(datatype, &type_size);

    //for every thread on every rank send to the root.

    int tid = omp_get_thread_num();
    int threads = omp_get_num_threads();

    for(int i = 0; i < threads; i++)
    {
        #pragma omp barrier
        if(tid == i)
            MPI_Reduce(send_data, rec_data, count, datatype, op, root, communicator);
        #pragma omp barrier  
    }

    return MPI_SUCCESS;
};

int batch_allreduce(const void* send_data, void* rec_data, const int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm communicator)
{
    int task;
    MPI_Comm_rank(communicator, &task);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int type_size;
    MPI_Type_size(datatype, &type_size);

    //for every thread on every rank send to the root.

    int tid = omp_get_thread_num();
    int threads = omp_get_num_threads();

    for(int i = 0; i < threads; i++)
    {
        #pragma omp barrier
        if(tid == i)
            MPI_Allreduce(send_data, rec_data, count, datatype, op, communicator);
        #pragma omp barrier  
    }

    return MPI_SUCCESS;
};