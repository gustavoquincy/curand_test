#include <cuda.h>
#include <mpi.h>
#include <iostream>

int main(int argc, char *argv[]) {
    int rank, ndevices;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cout << "rank is: " << rank << std::endl;

    MPI_Finalize();
    return 0;
}
