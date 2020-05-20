
#include <graphlab.hpp>
int main(int argc, char** argv) {
{
    volatile int i = 1;
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("PID %d on %s ready for attach\n", getpid(), hostname);
    fflush(stdout);
    while (0 == i)
        sleep(5);
}
	// Initialize control plain using mpi
	//graphlab::mpi_tools::init(argc, argv);
const int required(MPI_THREAD_SINGLE);
int provided(-1);
int error = MPI_Init_thread(&argc, &argv, required, &provided);
assert(provided >= required);
assert(error == MPI_SUCCESS);
    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // Print off a hello world message
    printf("Hello world from processor %s, rank %d out of %d processors\n",
           processor_name, world_rank, world_size);
	graphlab::distributed_control dc;

    // Finalize the MPI environment.
    MPI_Finalize();


	//global_logger().set_log_level(LOG_INFO);
	//k
	return 0;
}
