//	Parallel and Distributed Programming: Final Project
//	Reinforcement Learning: A Parallel Implementation using MPI
//	Implementation using Blocking Communication
//	Aleksandra Obeso Duque

// mpicc -O3 -o rl_mpi rl_mpi.c -lmpi -lm
// mpirun -np 1 ./rl_mpi 1000 1000 1 0.99 1 0.0000001 100000000 1000 1 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>

# define PRINT_RESULT 0

typedef enum {
	NORTH,
	SOUTH,
	WEST,
	EAST,
} action;

typedef enum {
	TOP_LEFT,
	TOP_RIGHT,
	BOTTOM_LEFT,
	BOTTOM_RIGHT,
	TOP_INNER,
	BOTTOM_INNER,
	LEFT_INNER,
	RIGHT_INNER,
	INNER,
	TOP_COL,
	BOTTOM_COL,
	INNER_COL,
	LEFT_ROW,
	RIGHT_ROW,
	INNER_ROW,
	UNIQUE
} sg_loc;

typedef struct {
	int m;
	int n;
} gridsize;

typedef struct {
	int x;
	int y;
} state;

typedef struct {
	double *north;
	double *south;
	double *west;
	double *east;
	double *conv;
	bool *conv_flag;
} qv;

typedef struct {
	int top_bnd;
	int bot_bnd;
	int left_bnd;
	int right_bnd;
} bnds;

typedef struct {
	bool left;
	bool right;
	bool top;
	bool bottom;
} neigh;

action get_max_action(qv qv_current, state agent_state, gridsize gs);
action e_greedy(double epsilon, qv qv_current, state agent_state, gridsize gs);
double get_qv(state agent_state, action opt, qv qv_current, gridsize gs);
void move_agent(state *agent_state, action opt, gridsize gs);
void update_qv(state agent_state, state agent_next_state, action opt, qv qvalues, double eta, double r, double gamma, gridsize gs, int *conv_counter, double max_error);
double get_max_qv_next(qv qvalue, state agent_state, gridsize gs);
void calc_convergence(double *qv_conv, bool trophy_flag, state trophy_state, double gamma, gridsize subgrid, bnds my_bnds, gridsize offset, double input_reward);
double timer();
sg_loc adapt_subgrid(int my_crds[], gridsize *subgrid, gridsize *sg_remain, bnds *my_bnds, gridsize *p_grid, gridsize *offset, neigh *my_neigh);
bool validate_state(state *agent_state, bnds my_bnds);

int main (int argc, char **argv) {

	// Check input arguments
	if (argc != 11) {
		printf("Please specify:\n"	\
			"\t     1, 2) m, n: Gridworld size [2, +oo)\n"	\
			"\t\t3) eta: Learning rate (0, 1]\n"	\
			"\t\t4) gamma: Discount factor (0, 1]\n"	\
			"\t\t5) epsilon: Exploration rate Greedy -> Random [0, 1]\n"	\
			"\t\t6) max_error: Convergence error\n" \
			"\t\t7) reward: Reward value [1, +oo)\n"	\
			"\t\t8) comm_steps: Number of time steps between processor communication [1, +oo)\n"	\
			"\t\t9, 10) px, py: Processor number\n");
		return -1;
	}
	
	// Get and validate input parameters
	gridsize grid;
	grid.m = atoi(argv[1]);
	grid.n = atoi(argv[2]);
	if (grid.m*grid.n < 2) {
		printf("m*n must be in the interval [2, +oo)!\n");
		return -1;
	}
	
	double eta = atof(argv[3]);
	if (eta <= 0 || eta > 1) {
		printf("eta must be in the interval (0, 1]!\n");
		return -1;
	}
	
	double gamma = atof(argv[4]);
	if (gamma <= 0 || gamma > 1) {
		printf("gamma must be in the interval (0, 1]!\n");
		return -1;
	}
	
	double epsilon = atof(argv[5]);
	if (epsilon < 0 || epsilon > 1) {
		printf("epsilon must be in the interval [0, 1]!\n");
		return -1;
	}
	
	double max_error = atof(argv[6]);
	if (epsilon < 0 || epsilon > 1) {
		printf("max_error must be in the interval [0, +oo)!\n");
		return -1;
	}
	
	double input_reward = atof(argv[7]);
	if (input_reward < 1) {
		printf("reward must be in the interval [1, +oo)!\n");
		return -1;
	}
	
	int comm_steps = atoi(argv[8]);
	if (comm_steps < 1) {
		printf("comm_steps must be in the interval [1, +oo)!\n");
		return -1;
	}
	
	gridsize p_grid;
	p_grid.m = atoi(argv[9]);
	p_grid.n = atoi(argv[10]);
	if (p_grid.m*p_grid.n < 1) {
		printf("px*py must be in the interval [1, +oo)!\n");
		return -1;
	}
	
	// MPI initialization
	MPI_Init(&argc, &argv);
	
	int p_grid_size;
	MPI_Comm_size(MPI_COMM_WORLD, &p_grid_size);
	
	if (p_grid_size != p_grid.m*p_grid.n) {
		printf("px*py must be equal to the specified number of processors!\n");
		return -1;
	}
	
	// Get global rank
	int my_glob_rank;
	int print_rank = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_glob_rank);
	
	// Create cartesian communicator
	MPI_Comm crt_comm;
	int ndims =  2;
	int dims[] = {p_grid.m, p_grid.n};
	int cyclic[] = {0, 0};
	int reorder = 0;
	MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, cyclic, reorder, &crt_comm);
	
	// Get cartesian rank and coords
	int my_cart_rank, my_crds[ndims];
	MPI_Comm_rank(crt_comm, &my_cart_rank);
	MPI_Cart_coords(crt_comm, my_cart_rank, ndims, my_crds);
	
	// Split cartesian communicator into columns and rows
	MPI_Comm row_comm, col_comm;
	MPI_Comm_split(crt_comm, my_crds[0], my_crds[1], &row_comm);
	MPI_Comm_split(crt_comm, my_crds[1], my_crds[0], &col_comm);
	
	gridsize subgrid, subgrid_const, sg_remain, offset;
	subgrid.m = grid.m / p_grid.m;
	sg_remain.m = grid.m % p_grid.m;
	subgrid.n = grid.n / p_grid.n;
	sg_remain.n = grid.n % p_grid.n;
		
	// Adjust non-uniform data partition
	if (my_crds[0] < sg_remain.m)
		subgrid.m += 1;
		
	if (my_crds[1] < sg_remain.n)
		subgrid.n += 1;
	
	subgrid_const = subgrid;
	
	if (subgrid_const.m < 4 || subgrid_const.n < 4) {
		printf("m/px and n/py must both be in the interval [4, +oo)!\n");
		return -1;
	}
	
	bnds my_bnds;
	neigh my_neigh;
	sg_loc my_loc = adapt_subgrid(my_crds, &subgrid, &sg_remain, &my_bnds, &p_grid, &offset, &my_neigh);
	
	double reward = 0.0;
	
	// Allocate and initialize Q-value matrices
	qv qv_current;
	
	qv_current.north = (double *) malloc(subgrid.m*subgrid.n*sizeof(double));
	qv_current.south = (double *) malloc(subgrid.m*subgrid.n*sizeof(double));
	qv_current.west = (double *) malloc(subgrid.m*subgrid.n*sizeof(double));
	qv_current.east = (double *) malloc(subgrid.m*subgrid.n*sizeof(double));
	qv_current.conv = (double *) malloc(subgrid.m*subgrid.n*sizeof(double));
	qv_current.conv_flag = (bool *) malloc(subgrid.m*subgrid.n*sizeof(bool));
	
	memset(qv_current.north, 0, subgrid.m*subgrid.n*sizeof(double));
	memset(qv_current.south, 0, subgrid.m*subgrid.n*sizeof(double));
	memset(qv_current.west, 0, subgrid.m*subgrid.n*sizeof(double));
	memset(qv_current.east, 0, subgrid.m*subgrid.n*sizeof(double));
	memset(qv_current.conv, 0, subgrid.m*subgrid.n*sizeof(double));
	memset(qv_current.conv_flag, false, subgrid.m*subgrid.n*sizeof(bool));
	
	// Get random coordinates for the trophy and agent
	bool trophy_flag = false;
	int counter = 0;
	state trophy_crds, trophy_state, trophy_state_glob, agent_state, agent_old_state;
	srand(time(NULL));
	
	trophy_crds.x = rand() % p_grid.m;
	trophy_crds.y = rand() % p_grid.n;
	
	int trophy_glob_rank = trophy_crds.x*p_grid.n + trophy_crds.y;
	
	if ((trophy_crds.x == my_crds[0]) && (trophy_crds.y == my_crds[1])) {
		trophy_flag = true;
		trophy_state.x = rand() % ((my_bnds.bot_bnd-1)+1-(my_bnds.top_bnd+1)) + (my_bnds.top_bnd+1);
		trophy_state.y = rand() % ((my_bnds.right_bnd-1)+1-(my_bnds.left_bnd+1)) + (my_bnds.left_bnd+1);
		
		trophy_state_glob.x = trophy_state.x - my_bnds.top_bnd + offset.m;
		trophy_state_glob.y = trophy_state.y - my_bnds.left_bnd + offset.n;
		
		qv_current.conv_flag[trophy_state.x*subgrid.n + trophy_state.y] = true;
		counter = 1;
	}
	
	if (p_grid.m*p_grid.n > 1)
		MPI_Bcast(&trophy_state_glob, 2, MPI_INT, trophy_glob_rank, MPI_COMM_WORLD);
	
	// Calculate convergence q-values
	calc_convergence(qv_current.conv, trophy_flag, trophy_state_glob, gamma, subgrid, my_bnds, offset, input_reward);
	int *conv_counter = &counter;
	
	int episodes = 1;
	
	double begin, end;
	long int k = 0;
	long int k_send = 0;
	
	// Create data type for row and col communication
	MPI_Datatype row_type, col_type;
	MPI_Type_vector(subgrid_const.m, 1, subgrid.n, MPI_DOUBLE, &col_type);
	MPI_Type_commit(&col_type);
	MPI_Type_vector(1, subgrid_const.n, subgrid.n, MPI_DOUBLE, &row_type);
	MPI_Type_commit(&row_type);
	
	// Create receive status
	MPI_Status recv_status;
	
	bool valid_state;
	
	bool local_converged = false;
	bool global_converged = false;
	
	begin = MPI_Wtime();
	while (!global_converged) {
		// Get random coordinates for the agent
		agent_state.x = rand() % (my_bnds.bot_bnd+1-my_bnds.top_bnd) + my_bnds.top_bnd;
		agent_state.y = rand() % (my_bnds.right_bnd+1-my_bnds.left_bnd) + my_bnds.left_bnd;
	
		// Same initial position must be forbidden
		while (trophy_flag && (trophy_state.x == agent_state.x) && (trophy_state.y == agent_state.y)) {
			agent_state.x = rand() % (my_bnds.bot_bnd+1-my_bnds.top_bnd) + my_bnds.top_bnd;
			agent_state.y = rand() % (my_bnds.right_bnd+1-my_bnds.left_bnd) + my_bnds.left_bnd;
		}
	
		// Q-learning algorithm
		action next_action = 0;
		double max_qv = 0.0;
		valid_state = true;
	
		// Selection an action based on epsilon greedy
		while (!(trophy_flag && (trophy_state.x == agent_state.x) && (trophy_state.y == agent_state.y)) &&	\
				!global_converged && valid_state) {
			next_action = e_greedy(epsilon, qv_current, agent_state, subgrid);
			max_qv = get_qv(agent_state, next_action, qv_current, subgrid);
			agent_old_state = agent_state;
			move_agent(&agent_state, next_action, subgrid);
			valid_state = validate_state(&agent_state, my_bnds);
		
			if (trophy_flag && (trophy_state.x == agent_state.x) && (trophy_state.y == agent_state.y))
				reward = input_reward;
			else
				reward = 0.0;
			update_qv(agent_old_state, agent_state, next_action, qv_current, eta, reward, gamma, subgrid, conv_counter, max_error);
			
			k++;
			k_send++;
			
			if (k_send == comm_steps) {
				// Row communication
				
				// Send north
				if (my_neigh.right) {
					MPI_Send(&qv_current.north[(my_bnds.top_bnd+1)*subgrid.n-2], 1, col_type, my_crds[1]+1, 0, row_comm);
				}
				
				if (my_neigh.left) {
					MPI_Recv(&qv_current.north[my_bnds.top_bnd*subgrid.n], 1, col_type, my_crds[1]-1, 0, row_comm, &recv_status);
					MPI_Send(&qv_current.north[my_bnds.top_bnd*subgrid.n+1], 1, col_type, my_crds[1]-1, 1, row_comm);
				}
				
				if (my_neigh.right) {
					MPI_Recv(&qv_current.north[(my_bnds.top_bnd+1)*subgrid.n-1], 1, col_type, my_crds[1]+1, 1, row_comm, &recv_status);
				}
				
				// Send south
				if (my_neigh.right) {
					MPI_Send(&qv_current.south[(my_bnds.top_bnd+1)*subgrid.n-2], 1, col_type, my_crds[1]+1, 2, row_comm);
				}
				
				if (my_neigh.left) {
					MPI_Recv(&qv_current.south[my_bnds.top_bnd*subgrid.n], 1, col_type, my_crds[1]-1, 2, row_comm, &recv_status);
					MPI_Send(&qv_current.south[my_bnds.top_bnd*subgrid.n+1], 1, col_type, my_crds[1]-1, 3, row_comm);
				}
				
				if (my_neigh.right) {
					MPI_Recv(&qv_current.south[(my_bnds.top_bnd+1)*subgrid.n-1], 1, col_type, my_crds[1]+1, 3, row_comm, &recv_status);
				}
				
				// Send west
				if (my_neigh.right) {
					MPI_Send(&qv_current.west[(my_bnds.top_bnd+1)*subgrid.n-2], 1, col_type, my_crds[1]+1, 4, row_comm);
				}
				
				if (my_neigh.left) {
					MPI_Recv(&qv_current.west[my_bnds.top_bnd*subgrid.n], 1, col_type, my_crds[1]-1, 4, row_comm, &recv_status);
					MPI_Send(&qv_current.west[my_bnds.top_bnd*subgrid.n+1], 1, col_type, my_crds[1]-1, 5, row_comm);
				}
				
				if (my_neigh.right) {
					MPI_Recv(&qv_current.west[(my_bnds.top_bnd+1)*subgrid.n-1], 1, col_type, my_crds[1]+1, 5, row_comm, &recv_status);
				}
				
				// Send east
				if (my_neigh.right) {
					MPI_Send(&qv_current.east[(my_bnds.top_bnd+1)*subgrid.n-2], 1, col_type, my_crds[1]+1, 6, row_comm);
				}
				
				if (my_neigh.left) {
					MPI_Recv(&qv_current.east[my_bnds.top_bnd*subgrid.n], 1, col_type, my_crds[1]-1, 6, row_comm, &recv_status);
					MPI_Send(&qv_current.east[my_bnds.top_bnd*subgrid.n+1], 1, col_type, my_crds[1]-1, 7, row_comm);
				}
				
				if (my_neigh.right) {
					MPI_Recv(&qv_current.east[(my_bnds.top_bnd+1)*subgrid.n-1], 1, col_type, my_crds[1]+1, 7, row_comm, &recv_status);
				}
				
				// Column communication
					
				// Send north
				if (my_neigh.bottom) {
					MPI_Send(&qv_current.north[subgrid.n*(subgrid.m-2)+my_bnds.left_bnd], 1, row_type, my_crds[0]+1, 8, col_comm);
				}
				
				if (my_neigh.top) {
					MPI_Recv(&qv_current.north[my_bnds.left_bnd], 1, row_type, my_crds[0]-1, 8, col_comm, &recv_status);
					MPI_Send(&qv_current.north[subgrid.n+my_bnds.left_bnd], 1, row_type, my_crds[0]-1, 9, col_comm);
				}
				
				if (my_neigh.bottom) {
					MPI_Recv(&qv_current.north[subgrid.n*(subgrid.m-1)+my_bnds.left_bnd], 1, row_type, my_crds[0]+1, 9, col_comm, &recv_status);
				}
				
				// Send south
				if (my_neigh.bottom) {
					MPI_Send(&qv_current.south[subgrid.n*(subgrid.m-2)+my_bnds.left_bnd], 1, row_type, my_crds[0]+1, 10, col_comm);
				}
				
				if (my_neigh.top) {
					MPI_Recv(&qv_current.south[my_bnds.left_bnd], 1, row_type, my_crds[0]-1, 10, col_comm, &recv_status);
					MPI_Send(&qv_current.south[subgrid.n+my_bnds.left_bnd], 1, row_type, my_crds[0]-1, 11, col_comm);
				}
				
				if (my_neigh.bottom) {
					MPI_Recv(&qv_current.south[subgrid.n*(subgrid.m-1)+my_bnds.left_bnd], 1, row_type, my_crds[0]+1, 11, col_comm, &recv_status);
				}
				
				// Send west
				if (my_neigh.bottom) {
					MPI_Send(&qv_current.west[subgrid.n*(subgrid.m-2)+my_bnds.left_bnd], 1, row_type, my_crds[0]+1, 12, col_comm);
				}
				
				if (my_neigh.top) {
					MPI_Recv(&qv_current.west[my_bnds.left_bnd], 1, row_type, my_crds[0]-1, 12, col_comm, &recv_status);
					MPI_Send(&qv_current.west[subgrid.n+my_bnds.left_bnd], 1, row_type, my_crds[0]-1, 13, col_comm);
				}
				
				if (my_neigh.bottom) {
					MPI_Recv(&qv_current.west[subgrid.n*(subgrid.m-1)+my_bnds.left_bnd], 1, row_type, my_crds[0]+1, 13, col_comm, &recv_status);
				}
				
				// Send east
				if (my_neigh.bottom) {
					MPI_Send(&qv_current.east[subgrid.n*(subgrid.m-2)+my_bnds.left_bnd], 1, row_type, my_crds[0]+1, 14, col_comm);
				}
				
				if (my_neigh.top) {
					MPI_Recv(&qv_current.east[my_bnds.left_bnd], 1, row_type, my_crds[0]-1, 14, col_comm, &recv_status);
					MPI_Send(&qv_current.east[subgrid.n+my_bnds.left_bnd], 1, row_type, my_crds[0]-1, 15, col_comm);
				}
				
				if (my_neigh.bottom) {
					MPI_Recv(&qv_current.east[subgrid.n*(subgrid.m-1)+my_bnds.left_bnd], 1, row_type, my_crds[0]+1, 15, col_comm, &recv_status);
				}
			
				k_send = 0;
			
				if (counter == subgrid_const.m*subgrid_const.n)
					local_converged = true;
			
				// Send convergence state
				MPI_Allreduce(&local_converged, &global_converged, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
			}
		}
		
		episodes++;
	}
	end = MPI_Wtime();
	
	if (my_cart_rank == print_rank) {
#if PRINT_RESULT	
		printf("Subgrid: m = %d, n = %d\n", subgrid.m, subgrid.n);
		printf("Trophy flag: %s\n", trophy_flag?"True":"False");
		printf("Trophy location: (%d, %d)\n", trophy_state.x, trophy_state.y);
		printf("Offset: (%d, %d)\n", offset.m, offset.n);
		printf("Boundaries: top = %d, bot = %d, left = %d, right = %d\n", my_bnds.top_bnd, my_bnds.bot_bnd, my_bnds.left_bnd, my_bnds.right_bnd);
	
		printf("North q-values:\n");
		for (int i=0; i<subgrid.m; ++i) {
			for (int j=0; j<subgrid.n; ++j) {
				printf("%lf ", qv_current.north[i*subgrid.n+j]);
			}
			printf("\n");
		}
	
		printf("South q-values:\n");
		for (int i=0; i<subgrid.m; ++i) {
			for (int j=0; j<subgrid.n; ++j) {
				printf("%lf ", qv_current.south[i*subgrid.n+j]);
			}
			printf("\n");
		}
	
		printf("West q-values:\n");
		for (int i=0; i<subgrid.m; ++i) {
			for (int j=0; j<subgrid.n; ++j) {
				printf("%lf ", qv_current.west[i*subgrid.n+j]);
			}
			printf("\n");
		}
	
		printf("East q-values:\n");
		for (int i=0; i<subgrid.m; ++i) {
			for (int j=0; j<subgrid.n; ++j) {
				printf	("%lf ", qv_current.east[i*subgrid.n+j]);
			}
			printf("\n");
		}
	
		printf("Convergence q-values:\n");
		for (int i=0; i<subgrid.m; ++i) {
			for (int j=0; j<subgrid.n; ++j) {
				printf("%lf ", qv_current.conv[i*subgrid.n+j]);
			}
			printf("\n");
		}
		
		printf("Converged states: %d\n", counter);
	
		printf("Episodes: %d\n", episodes);
#endif
		printf("Epochs: %ld\n", k);
	
		printf("Time elapsed: %g s\n", (end-begin));
	}
	
	free(qv_current.north);
	free(qv_current.south);
	free(qv_current.west);
	free(qv_current.east);
	free(qv_current.conv);
	free(qv_current.conv_flag);
	
	MPI_Finalize();
	
	return 0;
}

action get_max_action (qv qv_current, state agent_state, gridsize gs) {
	action max_act_vert = NORTH, max_act_horiz = WEST;
	double max_qv_vert = 0.0, max_qv_horiz = 0.0;
	
	if (qv_current.north[agent_state.x*gs.n + agent_state.y] > qv_current.south[agent_state.x*gs.n + agent_state.y]) {
		max_act_vert = NORTH;
		max_qv_vert = qv_current.north[agent_state.x*gs.n + agent_state.y];
	} else if (qv_current.north[agent_state.x*gs.n + agent_state.y] < qv_current.south[agent_state.x*gs.n + agent_state.y]) {
		max_act_vert = SOUTH;
		max_qv_vert = qv_current.south[agent_state.x*gs.n + agent_state.y];
	} else {
		max_act_vert = rand()%2;
		max_qv_vert = (max_act_vert==NORTH)?qv_current.north[agent_state.x*gs.n + agent_state.y]:qv_current.south[agent_state.x*gs.n + agent_state.y];
	}
	
	if (qv_current.west[agent_state.x*gs.n + agent_state.y] > qv_current.east[agent_state.x*gs.n + agent_state.y]) {
		max_act_horiz = WEST;
		max_qv_horiz = qv_current.west[agent_state.x*gs.n + agent_state.y];
	} else if (qv_current.west[agent_state.x*gs.n + agent_state.y] < qv_current.east[agent_state.x*gs.n + agent_state.y]) {
		max_act_horiz = EAST;
		max_qv_horiz = qv_current.east[agent_state.x*gs.n + agent_state.y];
	} else {
		max_act_horiz = rand()% 2+2;
		max_qv_horiz = (max_act_vert==WEST)?qv_current.west[agent_state.x*gs.n + agent_state.y]:qv_current.east[agent_state.x*gs.n + agent_state.y];
	}
	
	if (max_qv_vert > max_qv_horiz)
		return max_act_vert;
	else if (max_qv_vert < max_qv_horiz)
		return max_act_horiz;
	else
		return rand()%4;
}

action e_greedy (double epsilon, qv qv_current, state agent_state, gridsize gs) {
	//srand(time(NULL));
	double r = (double) rand() / (double) RAND_MAX;
	
	if (r<epsilon)
		return rand()%4; // select random action
	else
		return get_max_action(qv_current, agent_state, gs); // select best action
}

double get_qv (state agent_state, action opt, qv qv_current, gridsize gs) {
	switch (opt) {
		case NORTH:
			return qv_current.north[agent_state.x*gs.n + agent_state.y];
		case SOUTH:
			return qv_current.south[agent_state.x*gs.n + agent_state.y];
		case WEST:
			return qv_current.west[agent_state.x*gs.n + agent_state.y];
		case EAST:
			return qv_current.east[agent_state.x*gs.n + agent_state.y];
	}
}

void move_agent (state *agent_state, action opt, gridsize gs) {
	switch (opt) {
		case NORTH:
			if (agent_state->x > 0)
				agent_state->x -= 1;
			break;
		case SOUTH:
			if (agent_state->x < gs.m-1)
				agent_state->x += 1;
			break;
		case WEST:
			if (agent_state->y > 0)
				agent_state->y -= 1;
			break;
		case EAST:
			if (agent_state->y < gs.n-1)
				agent_state->y += 1;
			break;
	}
}

double get_max_qv_next (qv qvalue, state agent_state, gridsize gs) {
	double max_qv_vert = 0.0, max_qv_horiz = 0.0;
	
	if (qvalue.north[agent_state.x*gs.n + agent_state.y] >= qvalue.south[agent_state.x*gs.n + agent_state.y])
		max_qv_vert = qvalue.north[agent_state.x*gs.n + agent_state.y];
	else
		max_qv_vert = qvalue.south[agent_state.x*gs.n + agent_state.y];
	
	if (qvalue.west[agent_state.x*gs.n + agent_state.y] >= qvalue.east[agent_state.x*gs.n + agent_state.y])
		max_qv_horiz = qvalue.west[agent_state.x*gs.n + agent_state.y];
	else
		max_qv_horiz = qvalue.east[agent_state.x*gs.n + agent_state.y];
	
	if (max_qv_vert >= max_qv_horiz)
		return max_qv_vert;
	else
		return max_qv_horiz;
}

// Q-learning: Q(s, a) <- Q(s, a) + eta * (r + gamma * max_a'(Q(s', a')) - Q(s, a))
void update_qv (state agent_state, state agent_next_state, action opt, qv qvalues, double eta, double r, double gamma, gridsize gs, int *conv_counter, double max_error) {

	switch (opt) {
		case NORTH:
			qvalues.north[agent_state.x*gs.n + agent_state.y] = qvalues.north[agent_state.x*gs.n + agent_state.y] + 
			eta*(r + gamma * get_max_qv_next(qvalues, agent_next_state, gs) - qvalues.north[agent_state.x*gs.n + agent_state.y]);
			
			if (!(qvalues.conv_flag[agent_state.x*gs.n + agent_state.y]) && (fabs(qvalues.north[agent_state.x*gs.n + agent_state.y]-qvalues.conv[agent_state.x*gs.n + agent_state.y])<=max_error)) {
				qvalues.conv_flag[agent_state.x*gs.n + agent_state.y] = true;
				(*conv_counter)++;
			}
			break;
		case SOUTH:
			qvalues.south[agent_state.x*gs.n + agent_state.y] = qvalues.south[agent_state.x*gs.n + agent_state.y] + 
			eta*(r + gamma * get_max_qv_next(qvalues, agent_next_state, gs) - qvalues.south[agent_state.x*gs.n + agent_state.y]);
			
			if (!(qvalues.conv_flag[agent_state.x*gs.n + agent_state.y]) && (fabs(qvalues.south[agent_state.x*gs.n + agent_state.y]-qvalues.conv[agent_state.x*gs.n + agent_state.y])<=max_error)) {
				qvalues.conv_flag[agent_state.x*gs.n + agent_state.y] = true;
				(*conv_counter)++;
			}
			break;
		case WEST:
			qvalues.west[agent_state.x*gs.n + agent_state.y] = qvalues.west[agent_state.x*gs.n + agent_state.y] + 
			eta*(r + gamma * get_max_qv_next(qvalues, agent_next_state, gs) - qvalues.west[agent_state.x*gs.n + agent_state.y]);
			
			if (!(qvalues.conv_flag[agent_state.x*gs.n + agent_state.y]) && (fabs(qvalues.west[agent_state.x*gs.n + agent_state.y]-qvalues.conv[agent_state.x*gs.n + agent_state.y])<=max_error)) {
				qvalues.conv_flag[agent_state.x*gs.n + agent_state.y] = true;
				(*conv_counter)++;
			}
			break;
		case EAST:
			qvalues.east[agent_state.x*gs.n + agent_state.y] = qvalues.east[agent_state.x*gs.n + agent_state.y] + 
			eta*(r + gamma * get_max_qv_next(qvalues, agent_next_state, gs) - qvalues.east[agent_state.x*gs.n + agent_state.y]);
			
			if (!(qvalues.conv_flag[agent_state.x*gs.n + agent_state.y]) && (fabs(qvalues.east[agent_state.x*gs.n + agent_state.y]-qvalues.conv[agent_state.x*gs.n + agent_state.y])<=max_error)) {
				qvalues.conv_flag[agent_state.x*gs.n + agent_state.y] = true;
				(*conv_counter)++;
			}
			break;
	}
}

void calc_convergence (double *qv_conv, bool trophy_flag, state trophy_state_glob, double gamma, gridsize subgrid, bnds my_bnds, gridsize offset, double input_reward) {
	
	int i_glob = offset.m;
	int j_glob = offset.n;
	
	for (int i = my_bnds.top_bnd; i <= my_bnds.bot_bnd; ++i) {
		for (int j = my_bnds.left_bnd; j <= my_bnds.right_bnd; ++j) {
			if (!trophy_flag)
				qv_conv[i*subgrid.n+j] = input_reward*pow(gamma, abs(trophy_state_glob.x-i_glob)+abs(trophy_state_glob.y-j_glob)-1);
			else if (trophy_flag && !((i_glob == trophy_state_glob.x) && (j_glob == trophy_state_glob.y)))
				qv_conv[i*subgrid.n+j] = input_reward*pow(gamma, abs(trophy_state_glob.x-i_glob)+abs(trophy_state_glob.y-j_glob)-1);
			j_glob++;
		}
		j_glob = offset.n;
		i_glob++;
	}
}

sg_loc adapt_subgrid(int my_crds[], gridsize *subgrid, gridsize *sg_remain, bnds *my_bnds, gridsize *p_grid, gridsize *offset, neigh *my_neigh) {
	sg_loc my_loc;
	
	if (my_crds[0] < sg_remain->m)
		offset->m = (subgrid->m)*my_crds[0];
	else
		offset->m = (subgrid->m)*my_crds[0] + (sg_remain->m);
		
	if (my_crds[1] < sg_remain->n)
		offset->n = (subgrid->n)*my_crds[1];
	else
		offset->n = (subgrid->n)*my_crds[1] + (sg_remain->n);
	
	if ((p_grid->m == 1) && (p_grid->n == 1)) {
		my_loc = UNIQUE;
		my_bnds->top_bnd = 0;
		my_bnds->bot_bnd = subgrid->m-1;
		my_bnds->left_bnd = 0;
		my_bnds->right_bnd = subgrid->n-1;
		my_neigh->left = false;
		my_neigh->right = false;
		my_neigh->top = false;
		my_neigh->bottom = false;
	} else if (p_grid->m == 1) {
		if (my_crds[1] == 0) {
			my_loc = LEFT_ROW;
			subgrid->n += 1;
			my_bnds->top_bnd = 0;
			my_bnds->bot_bnd = subgrid->m-1;
			my_bnds->left_bnd = 0;
			my_bnds->right_bnd = subgrid->n-2;
			my_neigh->left = false;
			my_neigh->right = true;
			my_neigh->top = false;
			my_neigh->bottom = false;
		} else if (my_crds[1] == p_grid->n-1) {
			my_loc = RIGHT_ROW;
			subgrid->n += 1;
			my_bnds->top_bnd = 0;
			my_bnds->bot_bnd = subgrid->m-1;
			my_bnds->left_bnd = 1;
			my_bnds->right_bnd = subgrid->n-1;
			my_neigh->left = true;
			my_neigh->right = false;
			my_neigh->top = false;
			my_neigh->bottom = false;
		} else {
			my_loc = INNER_ROW;
			subgrid->n += 2;
			my_bnds->top_bnd = 0;
			my_bnds->bot_bnd = subgrid->m-1;
			my_bnds->left_bnd = 1;
			my_bnds->right_bnd = subgrid->n-2;
			my_neigh->left = true;
			my_neigh->right = true;
			my_neigh->top = false;
			my_neigh->bottom = false;
		}
	} else if (p_grid->n == 1) {
		if (my_crds[0] == 0) {
			my_loc = TOP_COL;
			subgrid->m += 1;
			my_bnds->top_bnd = 0;
			my_bnds->bot_bnd = subgrid->m-2;
			my_bnds->left_bnd = 0;
			my_bnds->right_bnd = subgrid->n-1;
			my_neigh->left = false;
			my_neigh->right = false;
			my_neigh->top = false;
			my_neigh->bottom = true;
		} else if (my_crds[0] == p_grid->m-1) {
			my_loc = BOTTOM_COL;
			subgrid->m += 1;
			my_bnds->top_bnd = 1;
			my_bnds->bot_bnd = subgrid->m-1;
			my_bnds->left_bnd = 0;
			my_bnds->right_bnd = subgrid->n-1;
			my_neigh->left = false;
			my_neigh->right = false;
			my_neigh->top = true;
			my_neigh->bottom = false;
		} else {
			my_loc = INNER_COL;
			subgrid->m += 2;
			my_bnds->top_bnd = 1;
			my_bnds->bot_bnd = subgrid->m-2;
			my_bnds->left_bnd = 0;
			my_bnds->right_bnd = subgrid->n-1;
			my_neigh->left = false;
			my_neigh->right = false;
			my_neigh->top = true;
			my_neigh->bottom = true;
		}
	} else if (my_crds[0] == 0 && my_crds[1] == 0) {
		my_loc = TOP_LEFT;
		subgrid->m += 1;
		subgrid->n += 1;
		my_bnds->top_bnd = 0;
		my_bnds->bot_bnd = subgrid->m-2;
		my_bnds->left_bnd = 0;
		my_bnds->right_bnd = subgrid->n-2;
		my_neigh->left = false;
		my_neigh->right = true;
		my_neigh->top = false;
		my_neigh->bottom = true;
	} else if (my_crds[0] == 0 && my_crds[1] == p_grid->n-1) {
		my_loc = TOP_RIGHT;
		subgrid->m += 1;
		subgrid->n += 1;
		my_bnds->top_bnd = 0;
		my_bnds->bot_bnd = subgrid->m-2;
		my_bnds->left_bnd = 1;
		my_bnds->right_bnd = subgrid->n-1;
		my_neigh->left = true;
		my_neigh->right = false;
		my_neigh->top = false;
		my_neigh->bottom = true;
	} else if (my_crds[0] == p_grid->m-1 && my_crds[1] == 0) {
		my_loc = BOTTOM_LEFT;
		subgrid->m += 1;
		subgrid->n += 1;
		my_bnds->top_bnd = 1;
		my_bnds->bot_bnd = subgrid->m-1;
		my_bnds->left_bnd = 0;
		my_bnds->right_bnd = subgrid->n-2;
		my_neigh->left = false;
		my_neigh->right = true;
		my_neigh->top = true;
		my_neigh->bottom = false;
	} else if (my_crds[0] == p_grid->m-1 && my_crds[1] == p_grid->n-1) {
		my_loc = BOTTOM_RIGHT;
		subgrid->m += 1;
		subgrid->n += 1;
		my_bnds->top_bnd = 1;
		my_bnds->bot_bnd = subgrid->m-1;
		my_bnds->left_bnd = 1;
		my_bnds->right_bnd = subgrid->n-1;
		my_neigh->left = true;
		my_neigh->right = false;
		my_neigh->top = true;
		my_neigh->bottom = false;
	} else if (my_crds[0] == 0) {
		my_loc = TOP_INNER;
		subgrid->m += 1;
		subgrid->n += 2;
		my_bnds->top_bnd = 0;
		my_bnds->bot_bnd = subgrid->m-2;
		my_bnds->left_bnd = 1;
		my_bnds->right_bnd = subgrid->n-2;
		my_neigh->left = true;
		my_neigh->right = true;
		my_neigh->top = false;
		my_neigh->bottom = true;
	} else if (my_crds[0] == p_grid->m-1) {
		my_loc = BOTTOM_INNER;
		subgrid->m += 1;
		subgrid->n += 2;
		my_bnds->top_bnd = 1;
		my_bnds->bot_bnd = subgrid->m-1;
		my_bnds->left_bnd = 1;
		my_bnds->right_bnd = subgrid->n-2;
		my_neigh->left = true;
		my_neigh->right = true;
		my_neigh->top = true;
		my_neigh->bottom = false;
	} else if (my_crds[1] == 0) {
		my_loc = LEFT_INNER;
		subgrid->m += 2;
		subgrid->n += 1;
		my_bnds->top_bnd = 1;
		my_bnds->bot_bnd = subgrid->m-2;
		my_bnds->left_bnd = 0;
		my_bnds->right_bnd = subgrid->n-2;
		my_neigh->left = false;
		my_neigh->right = true;
		my_neigh->top = true;
		my_neigh->bottom = true;
	} else if (my_crds[1] == p_grid->n-1) {
		my_loc = RIGHT_INNER;
		subgrid->m += 2;
		subgrid->n += 1;
		my_bnds->top_bnd = 1;
		my_bnds->bot_bnd = subgrid->m-2;
		my_bnds->left_bnd = 1;
		my_bnds->right_bnd = subgrid->n-1;
		my_neigh->left = true;
		my_neigh->right = false;
		my_neigh->top = true;
		my_neigh->bottom = true;
	} else {
		my_loc = INNER;
		subgrid->m += 2;
		subgrid->n += 2;
		my_bnds->top_bnd = 1;
		my_bnds->bot_bnd = subgrid->m-2;
		my_bnds->left_bnd = 1;
		my_bnds->right_bnd = subgrid->n-2;
		my_neigh->left = true;
		my_neigh->right = true;
		my_neigh->top = true;
		my_neigh->bottom = true;
	}
	
	return my_loc;
}

bool validate_state (state *agent_state, bnds my_bnds) {
	return ((agent_state->x >= my_bnds.top_bnd) && (agent_state->x <= my_bnds.bot_bnd)	\
		&& (agent_state->y >= my_bnds.left_bnd) && (agent_state->y <= my_bnds.right_bnd))	\
		?true:false;
}
