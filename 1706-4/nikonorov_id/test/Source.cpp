#include <time.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <mpi.h>
#include <cmath>
#include <limits>
using namespace std;
double sequential_code(double* x, int N)
{
	double sum = 0;
	for (int i = 0; i < N; i++)
	{
		sum += x[i];
	}
	return sum;
}
bool is_equal(double x, double y) 
{
	return fabs(x - y) < numeric_limits<double>::epsilon();
}

void check(int N, double res1, double res2)
{
	if (!is_equal(res1, res2))
	{
		printf("%g seq\n", res1);
		printf("%g mpi\n", res2);
		printf("Failed math\n");
	}
	return;
}
int main()
{
	double start; 
	double* arr = NULL;
	double* mas = NULL;
	double* s_mas = NULL;
	double sum, sum_all, res;
	int ProcNum, ProcRank;
	int N1 = 0;
    int N = 50000000;
	srand(time(NULL));
	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
	MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
	if (ProcRank == 0)
	{
		arr = new double[N];
		s_mas = new double[N];
		for (int i = 0; i < N; i++)
		{
			s_mas[i] = arr[i] = rand();
		}
		res = sequential_code(s_mas, N);
	}
	MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
	N1 = N / ProcNum;
	mas = new double[N1];
	MPI_Scatter(arr, N1, MPI_DOUBLE, mas, N1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	start = MPI_Wtime();
	sum = 0;
	sum_all = 0;
	for (int i = 0; i < N1; i++)
	{
		sum += mas[i];
	}
	if (ProcRank == 0)
	{
		for (int i = N1 * ProcNum; i < N; i++)
		{
			sum += arr[i];
		}
	}
	MPI_Reduce(&sum, &sum_all, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	if (ProcRank == 0)
	{
		printf("Result=%10.2f\n", sum_all);
		printf("Time=%f\n", (MPI_Wtime() - start));
		check(N, sum_all, res);
	}
		
	MPI_Finalize();
	delete[] arr;
	delete[] mas;
	delete[] s_mas;
	return 0;
}