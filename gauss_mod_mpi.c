#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <assert.h>
#include <mpi.h>

// Tamanho do bloco de pivôs que data processo vai finalizar antes de se comunicar
// com o próximo processo no pipeline
#define BLOCK_SIZE (getenv("BLOCK_SIZE") ? atoi(getenv("BLOCK_SIZE")) : 20)

// Exibe mensagens no console se `DEBUG=1`
#define DEBUG(fmt, args...)                                         \
	if (getenv("DEBUG") != NULL && strcmp(getenv("DEBUG"), "1") == 0) \
		printf("PROC(%d): " fmt "\n", procidx, ##args);

// Exibe mensagem na saída stderr (se for o processo `0`) e finaliza o programa
#define RTHROW(fmt, args...)           \
	if (procidx == 0)                    \
		fprintf(stderr, fmt "\n", ##args); \
	MPI_Finalize();                      \
	exit(EXIT_FAILURE);

void saveResult(double *A, double *b, double *x, int n);
int testLinearSystem(double *A, double *b, double *x, int n);
void loadLinearSystem(int n, double *A, double *b);
void solveLinearSystem(double *local_A, double *local_b, double *x, int n, int local_n, int procidx, int totalprocs);

int main(int argc, char **argv)
{
	int n, local_n, procidx, totalprocs, base, rem;
	double *A, *b, *x;
	double *local_A, *local_b;
	double t_total;
	double t_load;
	double t_comm;
	double t_solve;
	double t_test;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &procidx);
	MPI_Comm_size(MPI_COMM_WORLD, &totalprocs);

	if (argc < 2)
	{
		RTHROW("Uso: %s <n>", argv[0]);
	}

	if (!(totalprocs == 2 || totalprocs == 4 || totalprocs == 8 || totalprocs == 16 || totalprocs == 32))
	{
		RTHROW("Número de processos (%d) inválido, suportados apenas: 2, 4, 8, 16 ou 32", totalprocs);
	}

	if (BLOCK_SIZE <= 0)
	{
		RTHROW("Valor de BLOCK_SIZE deve ser positivo");
	}

	n = atoi(argv[argc - 1]);

	if (procidx == 0)
	{
		t_total = MPI_Wtime();
	}

	base = n / totalprocs;
	rem = n % totalprocs;
	local_n = base + (procidx < rem ? 1 : 0);
	local_A = (double *)malloc(local_n * n * sizeof(double));
	local_b = (double *)malloc(local_n * sizeof(double));

	if (procidx == 0)
	{
		DEBUG("Inicializando (totalprocs=%d, base=%d, rem=%d, BLOCK_SIZE=%d)", totalprocs, base, rem, BLOCK_SIZE);

		A = (double *)malloc(n * n * sizeof(double));
		b = (double *)malloc(n * sizeof(double));
		x = (double *)malloc(n * sizeof(double));
		t_load = MPI_Wtime();
		loadLinearSystem(n, A, b);
		t_load = MPI_Wtime() - t_load;
	}

	int *sendcounts_A = NULL, *displs_A = NULL, *sendcounts_b = NULL, *displs_b = NULL, p, rows_p, start_p;
	if (procidx == 0)
	{
		sendcounts_A = (int *)malloc(totalprocs * sizeof(int));
		displs_A = (int *)malloc(totalprocs * sizeof(int));
		sendcounts_b = (int *)malloc(totalprocs * sizeof(int));
		displs_b = (int *)malloc(totalprocs * sizeof(int));

		for (p = 0; p < totalprocs; p++)
		{
			rows_p = base + (p < rem ? 1 : 0);
			start_p = (p < rem) ? (p * (base + 1)) : (rem * (base + 1) + (p - rem) * base);
			sendcounts_A[p] = rows_p * n;
			displs_A[p] = start_p * n;
			sendcounts_b[p] = rows_p;
			displs_b[p] = start_p;
		}
	}

	// Distribui as linhas da matriz entre os processos. Caso o número de linhas não seja múltiplo
	// do número de processos, alguns processos receberam local_n + 1, enquanto o restante somente local_n
	// Assim distribuímos de forma uniforme o resto das linhas entre os processos
	if (procidx == 0)
		t_comm = MPI_Wtime();
	MPI_Scatterv((procidx == 0) ? A : NULL, sendcounts_A, displs_A, MPI_DOUBLE,
							 local_A, local_n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatterv((procidx == 0) ? b : NULL, sendcounts_b, displs_b, MPI_DOUBLE,
							 local_b, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	if (procidx == 0)
		t_comm = MPI_Wtime() - t_comm;

	if (procidx == 0)
		t_solve = MPI_Wtime();
	solveLinearSystem(local_A, local_b, x, n, local_n, procidx, totalprocs);
	if (procidx == 0)
		t_solve = MPI_Wtime() - t_solve;

	if (procidx == 0)
	{
		free(sendcounts_A);
		free(displs_A);
		free(sendcounts_b);
		free(displs_b);
	}

	// Garante que todos os processos finalizaram antes de exibir os resultados finais
	// O `solveLinearSystem` já fará isso, então essa barreira é só como garantia de que
	// as mensagens de resultado aparecerão por último
	MPI_Barrier(MPI_COMM_WORLD);

	if (procidx == 0)
	{
		t_test = MPI_Wtime();
		int nerros = testLinearSystem(A, b, x, n);
		t_test = MPI_Wtime() - t_test;

		t_total = MPI_Wtime() - t_total;

		printf("\n==== Resultados Finais ====\n");
		printf("Tempo total de carregamento = %.6f segundos (%.1f%%)\n", t_load, (t_load / t_total) * 100);
		printf("Tempo total de comunicação inicial = %.6f segundos (%.1f%%)\n", t_comm, (t_comm / t_total) * 100);
		printf("Tempo total solveLinearSystem = %.6f segundos (%.1f%%)\n", t_solve, (t_solve / t_total) * 100);
		printf("Tempo total testLinearSystem = %.6f segundos (%.1f%%)\n", t_test, (t_test / t_total) * 100);
		printf("Tempo total de execução = %.6f segundos\n", t_total);
		printf("Número de erros = %d\n", nerros);

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
		if (fabs(sum - b[i]) >= 0.001)
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
	DEBUG("Iniciando solveLinearSystem");

	// Variáveis de tempo para este processo
	double t_total; // Tempo total do processo
	double t_start_local = MPI_Wtime(), t_end_local;
	double t_comm = 0.0;				 // Tempo gasto em comunicação
	double t_comp = 0.0;				 // Tempo gasto em computação
	double t_idle = 0.0;				 // Tempo ocioso no pipeline (load imbalance)
	double t_start_op, t_end_op; // Tempos temporários para operações

	int pivot_row, pivot_owner, local_pivot_row, current_row_start, current_row, count,
			pivot_unit, local_pivot_offset, local_pivot_buffer_mat_size, block_tag = 0, start_owner;
	double pivot, b_pivot, ratio;

	// Buffer para armazenar a linha de pivô + elemento b
	// Usamos uma "matriz" para armazenar BLOCK_SIZE por vez para reduzir a quantidade de comunicações
	// Assim, cada processo calculará primeiro BLOCK_SIZE pivôs antes de passar o trabalho para
	// os próximos processos
	int pivot_buffer_size = n + 1;															// Tamanho do buffer para 1 pivô
	int pivot_buffer_mat_size = pivot_buffer_size * BLOCK_SIZE; // Tamanho da matriz do buffer de pivô
	double *pivot_buffer = (double *)malloc(pivot_buffer_mat_size * sizeof(double));

	MPI_Status status;

	/* Compute base and remainder once; we'll derive owner and starts arithmetically
		 This avoids allocating rows/starts arrays and a while-loop per pivot. */
	int base = n / totalprocs;
	int rem = n % totalprocs;
	int threshold = (base + 1) * rem; /* first 'rem' processes have (base+1) rows */

	/* Gaussian Elimination */
	for (pivot_row = 0; pivot_row < (n - 1); pivot_row++)
	{
		pivot_owner = pivot_row < threshold ? (pivot_row / (base + 1)) : rem + (pivot_row - threshold) / base;
		start_owner = (pivot_owner < rem) ? (pivot_owner * (base + 1)) : (rem * (base + 1) + (pivot_owner - rem) * base);
		local_pivot_row = pivot_row - start_owner;
		// Calcula pivot_unit relativo ao início do processo dono
		pivot_unit = local_pivot_row % BLOCK_SIZE;
		local_pivot_offset = pivot_unit * pivot_buffer_size;

		if (procidx == pivot_owner)
		{
			local_pivot_buffer_mat_size = pivot_buffer_size * (pivot_unit + 1);

			// O pivô está dentro da área deste processo, então copiamos a linha local para o buffer do pivô
			memcpy(&pivot_buffer[local_pivot_offset], &local_A[local_pivot_row * n], n * sizeof(double));
			pivot_buffer[local_pivot_offset + n] = local_b[local_pivot_row];

			// Envio para o "próximo" processo (se houver), a cada BLOCK_SIZE ou se for a última linha
			if (procidx < totalprocs - 1 && (pivot_unit == BLOCK_SIZE - 1 || local_pivot_row == local_n - 1))
			{
				DEBUG(
						"ENVIANDO pivô %d..%d para PROC(%d) (tag=%d, pivot_unit=%d, size=%d)",
						(pivot_row - pivot_unit), pivot_row, procidx + 1, block_tag, pivot_unit, local_pivot_buffer_mat_size);

				// Usamos 'pivot_row' como tag da mensagem
				t_start_op = MPI_Wtime();
				MPI_Send(pivot_buffer, local_pivot_buffer_mat_size, MPI_DOUBLE, procidx + 1, block_tag, MPI_COMM_WORLD);
				t_end_op = MPI_Wtime();
				t_comm += t_end_op - t_start_op;
				block_tag++;
			}
		}
		else if (procidx > pivot_owner)
		{
			// Podemos sempre esperar ler no inicio de um bloco, pois se o processo anterior não tiver um bloco inteiro para
			// processar, esse processo estará esperando igual e receberá o bloco parcial
			if (pivot_unit == 0)
			{
				MPI_Probe(procidx - 1, block_tag, MPI_COMM_WORLD, &status);
				MPI_Get_count(&status, MPI_DOUBLE, &local_pivot_buffer_mat_size);

				DEBUG(
						"RECEBENDO pivô %d..%d de PROC(%d) (tag=%d, pivot_unit=%d, size=%d)",
						pivot_row, pivot_row + local_pivot_buffer_mat_size / pivot_buffer_size - 1, procidx - 1, block_tag, pivot_unit,
						local_pivot_buffer_mat_size);

				// Este processo está "abaixo" do processo dono do pivô, então recebe as informações do pivô do
				// processo anterior
				t_start_op = MPI_Wtime();
				MPI_Recv(pivot_buffer, local_pivot_buffer_mat_size, MPI_DOUBLE, procidx - 1, block_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				t_end_op = MPI_Wtime();
				t_comm += t_end_op - t_start_op;

				// Reenvio para o "próximo" processo (se eu não for o último)
				if (procidx < totalprocs - 1)
				{
					t_start_op = MPI_Wtime();
					MPI_Send(pivot_buffer, local_pivot_buffer_mat_size, MPI_DOUBLE, procidx + 1, block_tag, MPI_COMM_WORLD);
					t_end_op = MPI_Wtime();
					t_comm += t_end_op - t_start_op;
				}

				block_tag++;
			}
		}
		else
		{
			// Pivô já está fora ("abaixo") da área desse processo, então não deve mais processar
			break;
		}

		pivot = pivot_buffer[local_pivot_offset + pivot_row];
		b_pivot = pivot_buffer[local_pivot_offset + n];

		// Se o pivô estiver dentro da área do processo atual, inicia a partir do pivô
		// Caso contrário, sempre inicia em 0 pois o processo estará computando sua
		// área para o pivô de um processo anterior
		current_row_start = procidx == pivot_owner ? local_pivot_row + 1 : 0;

		// Processa somente até o final da área atribuída a esse processo
		for (current_row = current_row_start; current_row < local_n; current_row++)
		{
			if (current_row == current_row_start)
				DEBUG("PROCESSANDO pivô %d da linha %d até %d", pivot_row, procidx * local_n + current_row_start, procidx * local_n + local_n);

			ratio = local_A[current_row * n + pivot_row] / pivot;
			for (count = pivot_row; count < n; count++)
			{
				local_A[current_row * n + count] -= (ratio * pivot_buffer[local_pivot_offset + count]);
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
	int *recvcounts_A = NULL, *recvdispls_A = NULL, *recvcounts_b = NULL, *recvdispls_b = NULL, p, rows_p, start_p;
	if (procidx == 0)
	{
		recvcounts_A = (int *)malloc(totalprocs * sizeof(int));
		recvdispls_A = (int *)malloc(totalprocs * sizeof(int));
		recvcounts_b = (int *)malloc(totalprocs * sizeof(int));
		recvdispls_b = (int *)malloc(totalprocs * sizeof(int));

		for (p = 0; p < totalprocs; p++)
		{
			rows_p = base + (p < rem ? 1 : 0);
			start_p = (p < rem) ? (p * (base + 1)) : (rem * (base + 1) + (p - rem) * base);
			recvcounts_A[p] = rows_p * n;
			recvdispls_A[p] = start_p * n;
			recvcounts_b[p] = rows_p;
			recvdispls_b[p] = start_p;
		}
	}

	// Primeiro sincroniza todos os processos para medir o tempo ocioso
	t_idle = MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);
	t_idle = MPI_Wtime() - t_idle;

	// Agora mede apenas o tempo efetivo de comunicação do Gatherv
	t_start_op = MPI_Wtime();
	MPI_Gatherv(local_A, local_n * n, MPI_DOUBLE,
							res_A, recvcounts_A, recvdispls_A, MPI_DOUBLE,
							0, MPI_COMM_WORLD);
	MPI_Gatherv(local_b, local_n, MPI_DOUBLE,
							res_b, recvcounts_b, recvdispls_b, MPI_DOUBLE,
							0, MPI_COMM_WORLD);
	t_end_op = MPI_Wtime();
	t_comm += t_end_op - t_start_op;

	if (procidx == 0)
	{
		free(recvcounts_A);
		free(recvdispls_A);
		free(recvcounts_b);
		free(recvdispls_b);
	}

	// Marca o tempo final
	t_end_local = MPI_Wtime();

	// Calcula o tempo total e de computação
	t_total = t_end_local - t_start_local;
	t_comp = t_total - t_comm - t_idle;

	printf("\nProcesso %d:\n", procidx);
	printf("  Tempo total = %.6f segundos\n", t_total);
	printf("  Tempo de computação = %.6f segundos (%.1f%%)\n", t_comp, (t_comp / t_total) * 100);
	printf("  Tempo de comunicação = %.6f segundos (%.1f%%)\n", t_comm, (t_comm / t_total) * 100);
	printf("  Tempo ocioso no pipeline (load imbalance) = %.6f segundos (%.1f%%)\n", t_idle, (t_idle / t_total) * 100);

	if (procidx != 0)
	{
		// Processos que não forem root não executam mais nada a partir daqui
		return;
	}

	// Processo 0 continua com back-substitution
	t_start_local = MPI_Wtime();

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

	// Calcula e exibe os tempos do processo 0
	t_end_local = MPI_Wtime();
	printf("\nTempo de execução do back-substitution = %.6f segundos\n", t_end_local - t_start_local);
}
