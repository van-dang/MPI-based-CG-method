cc poisson3d.c -o poisson3d -lm
#!/bin/bash                                                                                                                                            
for n in $(seq 1 64); do
    export MPI_NUM_THREADS=$n
    # echo MPI_NUM_THREADS=$n
    aprun -n $MPI_NUM_THREADS ./poisson3d C 512 512 512
done

