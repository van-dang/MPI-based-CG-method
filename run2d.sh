cc poisson2d.c -o poisson2d -lm
#!/bin/bash                                                                                                                                            
for n in $(seq 1 64); do
    export MPI_NUM_THREADS=$n
    # echo MPI_NUM_THREADS=$n
    aprun -n $MPI_NUM_THREADS ./poisson2d C 256 256 
done

