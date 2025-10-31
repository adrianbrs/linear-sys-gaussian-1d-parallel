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
void solveLinearSystem(double *local_A, double *local_b, double *x, int n, int local_n, int procidx, int totalprocs);

int main(int argc, char **argv)
{
	int n, local_n, procidx, totalprocs;
	double *A, *b, *x;
	double *local_A, *local_b;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &procidx);
	MPI_Comm_size(MPI_COMM_WORLD, &totalprocs);

	if (argc < 2)
	{
		if (procidx == 0)
		{
			fprintf(stderr, "Uso: %s [options] <n>\n", argv[0]);
		}
		return EXIT_FAILURE;
	}

	if (!(totalprocs == 2 || totalprocs == 4 || totalprocs == 8 || totalprocs == 16 || totalprocs == 32))
	{
		if (procidx == 0)
		{
			fprintf(stderr, "Número de processos (%d) inválido, suportados apenas: 2, 4, 8, 16 ou 32", totalprocs);
		}
		return EXIT_FAILURE;
	}

	n = atoi(argv[argc - 1]);
	local_n = n / totalprocs;
	local_A = (double *)malloc(local_n * n * sizeof(double));
	local_b = (double *)malloc(local_n * sizeof(double));

	if (procidx == 0)
	{
		A = (double *)malloc(n * n * sizeof(double));
		b = (double *)malloc(n * sizeof(double));
		x = (double *)malloc(n * sizeof(double));
		loadLinearSystem(n, A, b);
	}

	MPI_Scatter(A, local_n * n, MPI_DOUBLE,
							local_A, local_n * n, MPI_DOUBLE,
							0, MPI_COMM_WORLD);

	MPI_Scatter(b, local_n, MPI_DOUBLE,
							local_b, local_n, MPI_DOUBLE,
							0, MPI_COMM_WORLD);

	solveLinearSystem(local_A, local_b, x, n, local_n, procidx, totalprocs);

	if (procidx == 0)
	{
		int nerros = testLinearSystem(A, b, x, n);
		printf("Errors=%d\n", nerros);
		saveResult(A, b, x, n);
		free(A);
		free(b);
		free(x);
	}

	free(local_A);
	free(local_b);

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

void solveLinearSystem(double *local_A, double *local_b, double *x, int n, int local_n, int procidx, int totalprocs)
{
	int pivot_row, pivot_owner, local_pivot_row, current_row_start, current_row, count;
	double pivot, b_pivot, ratio;

	// Buffer para armazenar a linha de pivô + elemento b
	double *pivot_buffer = (double *)malloc((n + 1) * sizeof(double));

	/* Gaussian Elimination */
	for (pivot_row = 0; pivot_row < (n - 1); pivot_row++)
	{
		pivot_owner = pivot_row / local_n;
		local_pivot_row = pivot_row % local_n;

		if (procidx == pivot_owner)
		{
			// O pivô está dentro da área deste processo, então copiamos a linha local para o buffer do pivô
			memcpy(pivot_buffer, &local_A[local_pivot_row * n], n * sizeof(double));
			pivot_buffer[n] = local_b[local_pivot_row];

			// Envio para o "próximo" processo (se houver)
			if (procidx < totalprocs - 1)
			{
				// Usamos 'pivot_row' como tag da mensagem
				MPI_Send(pivot_buffer, n + 1, MPI_DOUBLE, procidx + 1, pivot_row, MPI_COMM_WORLD);
			}
		}
		else if (procidx > pivot_owner)
		{
			// Este processo está "abaixo" do processo dono do pivô, então recebe as informações do pivô do
			// processo anterior
			MPI_Recv(pivot_buffer, n + 1, MPI_DOUBLE, procidx - 1, pivot_row, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			// Reenvio para o "próximo" processo (se eu não for o último)
			if (procidx < totalprocs - 1)
			{
				MPI_Send(pivot_buffer, n + 1, MPI_DOUBLE, procidx + 1, pivot_row, MPI_COMM_WORLD);
			}
		}
		else
		{
			// Pivô já está fora ("abaixo") da área desse processo, então não deve mais processar
			// Ficará ocioso a partir de agora esperando os outros processos concluírem suas áreas
			break;
		}

		pivot = pivot_buffer[pivot_row];
		b_pivot = pivot_buffer[n];

		// Se o pivô estiver dentro da área do processo atual, inicia a partir do pivô
		// Caso contrário, sempre inicia em 0 pois o processo estará computando sua
		// área para o pivô de um processo anterior
		current_row_start = procidx == pivot_owner ? local_pivot_row + 1 : 0;

		// Processa somente até o final da área atribuída a esse processo
		for (current_row = current_row_start; current_row < local_n; current_row++)
		{
			ratio = local_A[current_row * n + pivot_row] / pivot;
			for (count = pivot_row; count < n; count++)
			{
				local_A[current_row * n + count] -= (ratio * pivot_buffer[count]);
			}
			local_b[current_row] -= (ratio * b_pivot);
		}
	}

	free(pivot_buffer);

	double *res_A, *res_b;

	if (procidx == 0)
	{
		// Aloca memória para o resultado final, agrupado de todos os processos, somente no processo root
		res_A = (double *)malloc(n * n * sizeof(double));
		res_b = (double *)malloc(n * sizeof(double));
	}

	// Agrupa novamente os valores calculados de todos os processos no processo root
	MPI_Gather(local_A, local_n * n, MPI_DOUBLE,
						 res_A, local_n * n, MPI_DOUBLE,
						 0, MPI_COMM_WORLD);
	MPI_Gather(local_b, local_n, MPI_DOUBLE,
						 res_b, local_n, MPI_DOUBLE,
						 0, MPI_COMM_WORLD);

	if (procidx != 0)
	{
		// A partir daqui, somente o processo 0 preciso calcular de forma sequencial
		return;
	}

	/* Back-substitution */
	x[n - 1] = res_b[n - 1] / res_A[(n - 1) * n + n - 1];
	for (pivot_row = (n - 2); pivot_row >= 0; pivot_row--)
	{
		double temp = res_b[pivot_row];
		for (current_row = (pivot_row + 1); current_row < n; current_row++)
		{
			temp -= (res_A[pivot_row * n + current_row] * x[current_row]);
		}
		x[pivot_row] = temp / res_A[pivot_row * n + pivot_row];
	}

	free(res_A);
	free(res_b);
}
