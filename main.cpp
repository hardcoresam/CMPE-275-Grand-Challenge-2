#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <assert.h>

using namespace std;

//This is the input matrices size
#define M_SIZE 10
#define MASTER_RANK 0

//Helper method which will give a random number
int GenerateRandomNumber() {
    return std::rand() % 9 + 1;
}

//Helper method which will fill our matrix with random numbers to create input matrices
void FillMatrix(int *A) {
    for (int i = 0; i < M_SIZE; i++) {
        for (int j = 0; j < M_SIZE; j++) {
            A[i * M_SIZE + j] = GenerateRandomNumber();
        }
    }
}

//Helper method to print the final calculated matrix
void PrintMatrix(int *matrix) {
    printf("\n");
    for (int i = 0; i < M_SIZE; i++) {
        for (int j = 0; j < M_SIZE; j++) {
            printf("%d ", matrix[i * M_SIZE + j]);
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {
    MPI_Init(NULL, NULL);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    //We are now splitting our entire world into 4 communication groups here
    int no_of_clusters = 4;
    int color = world_rank % no_of_clusters;
    MPI_Comm my_comm;

    //We are using MPI_Comm_split below to create different communication groups
    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &my_comm);
    int local_size, local_rank;
    MPI_Comm_size(my_comm, &local_size);
    MPI_Comm_rank(my_comm, &local_rank);

    int *a = (int *) malloc(M_SIZE * M_SIZE * sizeof(int));
    int *b = (int *) malloc(M_SIZE * M_SIZE * sizeof(int));
    int *c = (int *) malloc(M_SIZE * M_SIZE * sizeof(int));
    int offset, rows_num, remainder, whole_part, message_tag, i, j, k;
    long long int startTime, endTime;

    if (world_rank == MASTER_RANK) {
        printf("Generating A matrix with size %dx%d\n", M_SIZE, M_SIZE);
        FillMatrix(a);

        printf("Generating B matrix with size %dx%d\n", M_SIZE, M_SIZE);
        FillMatrix(b);

        printf("Starting multiplication of A and B matrices...\n");
        startTime = clock();

        whole_part = M_SIZE / no_of_clusters;
        remainder = M_SIZE % no_of_clusters;
        offset = 0;

        //Here, we are splitting our 'a' matrix and sending the divided matrices to different communication groups
        for (int group = 0; group < no_of_clusters; group++) {
            rows_num = group <= remainder ? whole_part + 1 : whole_part;
            MPI_Send(&offset, 1, MPI_INT, group, 0, MPI_COMM_WORLD);
            MPI_Send(&rows_num, 1, MPI_INT, group, 0, MPI_COMM_WORLD);
            MPI_Send(&a[offset * M_SIZE + 0], rows_num * M_SIZE, MPI_INT, group, 0, MPI_COMM_WORLD);
            MPI_Send(&b, M_SIZE * M_SIZE, MPI_INT, group, 0, MPI_COMM_WORLD);
            offset += rows_num;
        }
    } else {
        //Here, the 0 process inside each communication group receives the divided matrices
        if (local_rank == 0) {
            MPI_Recv(&offset, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&rows_num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&a, rows_num * M_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&b, M_SIZE * M_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    int rows_per_local_process = rows_num / local_size;
    int *sub_matrix = (int *) malloc(rows_per_local_process * rows_per_local_process * sizeof(int));
    int *sub_matrix_result = (int *) malloc(rows_per_local_process * rows_per_local_process * sizeof(int));
    int *sub_matrix_of_c_matrix = (int *) malloc(rows_per_local_process * rows_per_local_process * sizeof(int));
    assert(sub_matrix != NULL);
    assert(sub_matrix_result != NULL);
    assert(sub_matrix_of_c_matrix != NULL);

    // Broadcast the required information to other processes in the communication group
    MPI_Bcast(&offset, 1, MPI_INT, 0, my_comm);
    MPI_Bcast(&rows_num, 1, MPI_INT, 0, my_comm);
    MPI_Bcast(&b, M_SIZE * M_SIZE, MPI_INT, 0, my_comm);

    // Scatter is used to scatter the "a" matrix with all the other processes in the communication group
    MPI_Scatter(&a, rows_per_local_process * rows_per_local_process, MPI_INT, sub_matrix,
                rows_per_local_process * rows_per_local_process, MPI_INT, 0, my_comm);


    //The matrix multiplication for the matrices goes here and we are storing the sub results in sub_matrix_result.
    for (k = 0; k < M_SIZE; k++) {
        for (i = 0; i < rows_num; i++) {
            for (j = 0; j < M_SIZE; j++) {
                sub_matrix_result[M_SIZE * i + k] += sub_matrix[M_SIZE * i + j] * b[M_SIZE * j + k];
            }
        }
    }

    //Here, we are gathering back the resulted sub matrices into the c sub matrix
    MPI_Gather(&sub_matrix_result, rows_per_local_process * rows_per_local_process, MPI_INT, &sub_matrix_of_c_matrix,
               rows_per_local_process * rows_per_local_process, MPI_INT, 0, my_comm);


    //Here, 0 process in each communication group is sending back the computed result to the Master rank process
    if (local_rank == 0) {
        MPI_Send(&sub_matrix_of_c_matrix, rows_per_local_process * rows_per_local_process, MPI_INT, 0, local_rank,
                 MPI_COMM_WORLD);
    }

    //Here, the master rank process will receive the sub computed matrices and display the final result
    if (world_rank == MASTER_RANK) {
        for (int group = 0; group < no_of_clusters; group++) {
            MPI_Recv(&c[M_SIZE * offset + 0], rows_num * M_SIZE, MPI_INT, group, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        printf("Finished computing the matrix multiplication of below matrices...\n");
        printf("'a' matrix is printed below...\n");
        PrintMatrix(a);
        printf("'b' matrix is printed below...\n");
        PrintMatrix(b);
        printf("Computed 'c' matrix after multiplication is printed below...\n");
        PrintMatrix(c);

        endTime = clock();
        double totalExecutionTime = (double) ((endTime - startTime) / (1.0 * 1000000));
        printf("Total execution time for multiplying 2 %dx%d matrices is %f", M_SIZE, M_SIZE, totalExecutionTime);
    }

    MPI_Finalize();
    return 0;
}