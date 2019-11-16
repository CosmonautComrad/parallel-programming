#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <mpi.h>
#include <math.h>

using namespace std;


int calc_depth(int ProcNum) 
{
	int res = 0;
	while (ProcNum != 1) 
	{
		ProcNum = ProcNum >> 1;
		res++;
	}
	return res;
}

void My_Bcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
{
	int currRank;
	int ProcNum;
	int depth;

	MPI_Status status;
	MPI_Comm_size(comm, &ProcNum);
	MPI_Comm_rank(comm, &currRank);
	depth = calc_depth(ProcNum);
	if (currRank == 0) 
	{
		cout << "Depth = " << depth << endl;
	}
	if (currRank == root && root != 0) 
	{
		MPI_Send(buffer, count, datatype, 0, 0, comm);
	}
	if (currRank == 0 && root != 0) 
	{
		MPI_Recv(buffer, count, datatype, root, 0, comm, &status);
	}
	MPI_Barrier(comm);
	for (int i = 0; i < depth; i++) 
	{
		int here = (pow(2, i));
		for (int j = 0; j < here; j++) 
		{
			MPI_Barrier(comm);
			if (currRank == j)
				MPI_Send(buffer, count, datatype, j + here, 0, comm);
			if (currRank == j + here)
				MPI_Recv(buffer, count, datatype, j, 0, comm, &status);
		}

	}
	
	int here = static_cast<int>(pow(2, depth));
	for (int i = 0; i < ProcNum - here; i++)
	{
		MPI_Barrier(comm);
		if (currRank == i) 
			MPI_Send(buffer, count, datatype, i + here, 0, comm);
		if (currRank == i + here)
			MPI_Recv(buffer, count, datatype, i, 0, comm, &status);

	}

}
void main(int argc, char* argv[])
{
	int ProcRank;
	int ProcNum = 0;
	float* test = nullptr;
	double start;
	int root;
	if (argc < 2)
	{
		root = 0;
	}
	else
	{
		root = atoi(argv[1]);
	}
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
	MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
	test = new float[3];

	if (ProcRank == root) 
	{
		test[0] = 3.33;
		test[1] = 9.99;
		test[2] = 12.85;
	}
	start = MPI_Wtime();
	My_Bcast(test, 3, MPI_FLOAT, root, MPI_COMM_WORLD);
	for (int i = 0; i < ProcNum; i++)
	{
		if (ProcRank == i)
		{
			printf("Process %i : %f\n", ProcRank, test[1]);
			if (i == 0)
			{
				printf("Mpi time = %f\n", (MPI_Wtime() - start));
			}
		}
	}
	MPI_Finalize();
}