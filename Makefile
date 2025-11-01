n?=2000
np?=4
bsize?=20

build:
	mpicc gauss_mod_mpi.c -Wall -o gauss_mpi

buildseq:
	gcc -Wall gauss_mod.c -o gauss

run:
	BLOCK_SIZE=$(bsize) mpirun -np $(np) --hostfile ./hosts ./gauss_mpi $(n)

debug:
	DEBUG=1 BLOCK_SIZE=$(bsize) mpirun -np $(np) --hostfile ./hosts ./gauss_mpi $(n)

runseq:
	echo "$(n)" | ./gauss