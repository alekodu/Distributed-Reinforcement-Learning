CC         =  mpicc
CCFLAGS    =  -O3
CCGFLAGS   =  -g
LIBS       =  -lmpi -lm

BINS= rl_mpi_b

rl_mpi_b: rl_mpi_b.c
	$(CC) $(CCFLAGS) -o $@ $^ $(LIBS)

clean:
	$(RM) $(BINS)

