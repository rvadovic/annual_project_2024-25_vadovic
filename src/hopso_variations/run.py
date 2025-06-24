# Modified run.py
from mpi4py import MPI
import sys
import os
import numpy as np
from costF_2q_IvaH2_qiskit import objective_function_1
from costF_8q_LiH import cost_fn_8qlih
from hopso_mpi4py_arctan import hopso as hopso_arctan
from hopso_kill_internode import hopso as hopso_kill_internode
from time import perf_counter

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Synchronize all processes
comm.Barrier()

# Print process information
print(f"Process {rank}/{size} ready")

# Define parameters
hp = [1, 1, 2*np.pi, 0.01]
num_particles = 20
runs = 1
dimension = 80
maxcut = 2.05
max_iterations = 10000
e_min = []
vectors = []
vel_mag = []
gbest = []

# Verify total tasks matches available processes
#if rank == 0:
#    total_tasks = runs * num_particles
#    if total_tasks != size:
#        print(f"Error: Required tasks ({total_tasks}) != Available processes ({size})")
#        comm.Abort(1)

# Another barrier before starting main computation
comm.Barrier()

# Run HOPSO
start_time = perf_counter()
hopso_arctan(cost_fn_8qlih, hp, num_particles, runs, dimension, maxcut, e_min, vectors, vel_mag, gbest, max_iterations)
end_time = perf_counter()
print(end_time - start_time)