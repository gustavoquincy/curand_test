#include <mpi.h>
#include "stdio.h"

int main(int argc, char** argv)
{

    MPI_Init(NULL, NULL);


    int PID;
    MPI_Comm_rank(MPI_COMM_WORLD, &PID);


    int number_of_processes;
    MPI_Comm_size(MPI_COMM_WORLD, &number_of_processes);


    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_length;
    MPI_Get_processor_name(processor_name, &name_length);


    printf("Hello MPI user: from process PID %d out of %d processes on machine %s\n", PID, number_of_processes, processor_name);


    MPI_Finalize();

    return 0;
}
