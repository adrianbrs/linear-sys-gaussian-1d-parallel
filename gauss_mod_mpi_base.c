#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <assert.h>
#include <mpi.h>

void saveResult(double *A, double *b, double *x, int n);
int testLinearSystem(double *A, double *b, double *x, int n);
void loadLinearSystem(int n, double *A, double *b);
void solveLinearSystem(const double *A, const double *b, double *x, int n);

int main(int argc, char **argv)
{
	int n, rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (argc < 2)
	{
		if (rank == 0)
		{
			fprintf(stderr, "Uso: %s [options] <n>\n", argv[0]);
		}
		return EXIT_FAILURE;
	}

	if (!(size == 2 || size == 4 || size == 8 || size == 16 || size == 32))
	{
		if (rank == 0)
		{
			fprintf(stderr, "Número de processos (%d) inválido, suportados apenas: 2, 4, 8, 16 ou 32", size);
		}
		return EXIT_FAILURE;
	}

	n = atoi(argv[argc - 1]);

	double *A, *b, *x;

	if (rank == 0)
	{
		A = (double *)malloc(n * n * sizeof(double));
		b = (double *)malloc(n * sizeof(double));

		loadLinearSystem(n, &A[0], &b[0]);
	}
	else
	{
		x = (double *)malloc(n * sizeof(double));
	}

	solveLinearSystem(&A[0], &b[0], &x[0], n);

	if (rank == 0)
	{
		int nerros = testLinearSystem(&A[0], &b[0], &x[0], n);

		printf("Errors=%d\n", nerros);

		saveResult(&A[0], &b[0], &x[0], n);
	}

	MPI_Finalize();
	return EXIT_SUCCESS;
}

void saveResult(double *A, double *b, double *x, int n)
{
	int i;
	FILE *res;

	res = fopen("result.out", "w");
	if (res == NULL)
	{
		printf("File result.out does not open\n");
		exit(1);
	}

	for (i = 0; i < n; i++)
	{
		fprintf(res, "%.6f\n", x[i]);
	}

	fclose(res);
}

int testLinearSystem(double *A, double *b, double *x, int n)
{
	int i, j, c = 0;
	double sum = 0;

	for (i = 0; i < n; i++)
	{
		sum = 0;
		for (j = 0; j < n; j++)
			sum += A[i * n + j] * x[j];
		if (abs(sum - b[i]) >= 0.001)
		{
			printf("%f\n", (sum - b[i]));
			c++;
		}
	}
	return c;
}

void loadLinearSystem(int n, double *A, double *b)
{
	int i, j;
	FILE *mat, *vet;

	mat = fopen("matrix.in", "r");
	if (mat == NULL)
	{
		printf("File matrix.in does not open\n");
		exit(1);
	}

	vet = fopen("vector.in", "r");
	if (vet == NULL)
	{
		printf("File vector.in does not open\n");
		exit(1);
	}

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
			fscanf(mat, "%lf", &A[i * n + j]);
	}

	for (i = 0; i < n; i++)
		fscanf(vet, "%lf", &b[i]);

	fclose(mat);
	fclose(vet);
}

void solveLinearSystem(const double *A, const double *b, double *x, int n)
{
	double *Acpy = (double *)malloc(n * n * sizeof(double));
	double *bcpy = (double *)malloc(n * sizeof(double));
	memcpy(Acpy, A, n * n * sizeof(double));
	memcpy(bcpy, b, n * sizeof(double));

	int i, j, count;

	/* Gaussian Elimination */
	for (i = 0; i < (n - 1); i++)
	{

		for (j = (i + 1); j < n; j++)
		{
			double ratio = Acpy[j * n + i] / Acpy[i * n + i];
			for (count = i; count < n; count++)
			{
				Acpy[j * n + count] -= (ratio * Acpy[i * n + count]);
			}
			bcpy[j] -= (ratio * bcpy[i]);
		}
	}

	/* Back-substitution */
	x[n - 1] = bcpy[n - 1] / Acpy[(n - 1) * n + n - 1];
	for (i = (n - 2); i >= 0; i--)
	{
		double temp = bcpy[i];
		for (j = (i + 1); j < n; j++)
		{
			temp -= (Acpy[i * n + j] * x[j]);
		}
		x[i] = temp / Acpy[i * n + i];
	}
}
