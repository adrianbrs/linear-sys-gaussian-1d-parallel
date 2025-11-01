#!/bin/bash

# --- Configuração de Segurança ---
set -e
set -o pipefail

# --- 1. Verificação de Parâmetros ---
# Este script precisa de 2 parâmetros: n e times
if [ "$#" -ne 2 ]; then
    echo "Uso: $0 <n> <times>"
    echo "Exemplo: ./benchmark_seq.sh 3000 5"
    exit 1
fi

N=$1
TIMES=$2

# --- 2. Definição dos Nomes de Arquivo ---
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_FILE="benchmark-seq-n${N}-t${TIMES}-${TIMESTAMP}.log"
JSON_FILE="benchmark_seq_n${N}.json"

echo "Iniciando benchmark sequencial..."
echo "Tamanho (n): $N"
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
    
    # Executa o comando 'make runseq' sobrescrevendo a variável 'n' do Makefile
    # Assumimos que a versão sequencial também imprime "DEBUG=1"
    # Captura stdout e stderr
    output=$(DEBUG=1 make runseq n=$N 2>&1)
    
    # --- Anexa a saída completa ao arquivo de log ---
    echo "--- Execução $i/$TIMES ---" >> "$LOG_FILE"
    echo "$output" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE" # Adiciona uma linha em branco
    
    # Captura "Tempo total solveLinearSystem"
    # Assumimos que o printf é idêntico ao da versão mpi [cite: 1, `printf("Tempo total solveLinearSystem = %.6f segundos (%.1f%%)\n", t_solve, (t_solve / t_total) * 100);`]
    solve_time=$(echo "$output" | grep "Tempo total solveLinearSystem" | awk '{print $5}')
    
    if [ -z "$solve_time" ]; then
        echo "Erro na execução $i: Não foi possível capturar o tempo 'solveLinearSystem'."
        echo "O executável 'gauss' [cite: 1, `buildseq:\n\tgcc -Wall gauss_mod.c -o gauss`] (de 'make buildseq') precisa ter as mesmas métricas de printf."
        echo "Veja $LOG_FILE para a saída completa."
        exit 2
    fi
    
    solve_times_list+=($solve_time)
done

echo "------------------------------------"
echo "Benchmark concluído. Calculando estatísticas..."

# --- 4. Cálculo de Média e Desvio Padrão com AWK ---
# Adicionado 'LC_NUMERIC=C' para corrigir o bug do separador decimal (ponto vs vírgula)
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

json_list=$(printf "%s," "${solve_times_list[@]}")
json_list="[${json_list%,}]"

cat << EOF > $JSON_FILE
{
  "parametros": {
    "np": 1,
    "n": $N,
    "BLOCK_SIZE": null,
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