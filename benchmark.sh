#!/bin/bash

# --- Configuração de Segurança ---
set -e
set -o pipefail

# --- 1. Verificação de Parâmetros ---
if [ "$#" -ne 4 ]; then
    echo "Uso: $0 <np> <n> <BLOCK_SIZE> <times>"
    echo "Exemplo: ./benchmark.sh 4 3000 20 5"
    exit 1
fi

NP=$1
N=$2
BLOCK_SIZE_PARAM=$3
TIMES=$4

# --- 2. Definição dos Nomes de Arquivo ---
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_FILE="benchmark-np${NP}-n${N}-b${BLOCK_SIZE_PARAM}-t${TIMES}-${TIMESTAMP}.log"
JSON_FILE="benchmark_np${NP}_n${N}_b${BLOCK_SIZE_PARAM}.json"

echo "Iniciando benchmark..."
echo "Processos (np): $NP"
echo "Tamanho (n): $N"
echo "BLOCK_SIZE: $BLOCK_SIZE_PARAM"
echo "Repetições (times): $TIMES"
echo "------------------------------------"
echo "Logs completos serão salvos em: $LOG_FILE"
echo "Resultados JSON salvos em: $JSON_FILE"
echo "------------------------------------"

# --- 3. Loop de Execução e Coleta de Dados ---
declare -a solve_times_list

# Limpa o arquivo de log para esta nova execução
> "$LOG_FILE"

for (( i=1; i<=$TIMES; i++ )); do
    echo "Executando $i/$TIMES..."
    echo "--- Execução $i/$TIMES ---" >> "$LOG_FILE"
    
    # Desabilita 'set -e' temporariamente
    set +e 
    
    # 1. Executa o mpirun
    # 2. Redireciona stderr para stdout (2>&1)
    # 3. Usa 'tee -a' para anexar a saída ao log EM TEMPO REAL (streaming)
    # 4. A saída de 'tee' (que é a mesma) é capturada pela variável 'output'
    output=$(BLOCK_SIZE=$BLOCK_SIZE_PARAM mpirun -np $NP --hostfile hosts ./gauss_mpi $N 2>&1 | tee -a "$LOG_FILE")
    
    # Captura o código de saída do mpirun (índice 0), não do tee (índice 1)
    # Isso funciona por causa do 'set -o pipefail' no topo do script
    execution_exit_code=${PIPESTATUS[0]} 
    
    # Reabilita 'set -e'
    set -e 
    
    # Adiciona uma linha em branco ao log para legibilidade,
    # independentemente do resultado
    echo "" >> "$LOG_FILE" 

    # --- Verificação de Erro da Execução ---
    if [ $execution_exit_code -ne 0 ]; then
        echo "Erro na execução $i: mpirun falhou com código de saída $execution_exit_code."
        echo "A saída (parcial ou completa) foi registrada em $LOG_FILE."
        exit 2 # Aborta o script de benchmark
    fi
    
    # --- Captura "Tempo total solveLinearSystem" ---
    solve_time=$(echo "$output" | grep "Tempo total solveLinearSystem" | awk '{print $5}')
    
    if [ -z "$solve_time" ]; then
        echo "Erro na execução $i: Não foi possível capturar o tempo 'solveLinearSystem'."
        echo "Veja $LOG_FILE para a saída completa do MPI."
        exit 2
    fi
    
    solve_times_list+=($solve_time)
done

echo "------------------------------------"
echo "Benchmark concluído. Calculando estatísticas..."

# --- 4. Cálculo de Média e Desvio Padrão com AWK ---
stats_output=$(printf "%s\n" "${solve_times_list[@]}" | LC_NUMERIC=C awk '
    {
        sum += $1;
        sumsq += $1 * $1;
    }
    END {
        n = NR; # Número de Repetições
        if (n > 0) {
            avg = sum / n;
            # Fórmula do desvio padrão populacional
            stddev = sqrt( (sumsq / n) - (avg * avg) );
            printf "%.6f\n%.6f\n", avg, stddev;
        } else {
            printf "0\n0\n";
        }
    }
')

average=$(echo "$stats_output" | head -n 1)
std_dev=$(echo "$stats_output" | tail -n 1)

echo "Média (solveLinearSystem): $average s"
echo "Desvio Padrão (solveLinearSystem): $std_dev s"

# --- 5. Geração do JSON ---

# Formata a lista de tempos do bash para uma lista JSON
json_list=$(printf "%s," "${solve_times_list[@]}")
json_list="[${json_list%,}]" # Remove a vírgula final

# Escreve o arquivo JSON
cat << EOF > $JSON_FILE
{
  "parametros": {
    "np": $NP,
    "n": $N,
    "BLOCK_SIZE": $BLOCK_SIZE_PARAM,
    "times": $TIMES
  },
  "log_file": "$LOG_FILE",
  "tempos_solve_s": $json_list,
  "estatisticas": {
    "media_s": "$average",
    "desvio_padrao_s": "$std_dev"
  }
}
EOF

echo "------------------------------------"
echo "Resultados JSON salvos em: $JSON_FILE"
