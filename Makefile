n?=2000
np?=4

build:
	mpicc gauss_mod_mpi.c -o gauss_mpi

buildseq:
	gcc gauss_mod.c -o gauss

run:
	mpirun -np $(np) ./gauss_mpi $(n)

runseq:
	echo "$(n)" | ./gauss