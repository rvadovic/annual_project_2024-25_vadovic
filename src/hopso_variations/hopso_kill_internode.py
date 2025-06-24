from mpi4py import MPI
import numpy as np
import h5py
import time
import matplotlib.pyplot as plt
from hopso_result_handler import save_to_hdf5

def hopso(cost_fn, hp, num_particles, runs, dimension, max_cut, e_min, vectors, vel_mag, gbest, max_iterations):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print(f"Process {rank} starting...")

    size = comm.Get_size()
    
    # Get node information
    node_name = MPI.Get_processor_name()
    
    if rank == 0:
        print(f"Total processes: {size}")
        print(f"Starting distribution across nodes...")
    
    # Broadcast initial parameters to all nodes
    total_tasks = runs * num_particles
    
    if size != total_tasks:
        if rank == 0:
            print(f"Warning: Number of cores ({size}) doesn't match total tasks ({total_tasks})")
            print("Adjusting number of particles per run...")
        
        # Adjust num_particles to match available cores
        num_particles = size // runs
        total_tasks = runs * num_particles
        
        if rank == 0:
            print(f"Adjusted to {num_particles} particles per run")
    start_time = time.time()
    
        # Create node-aware communicator
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED, 0)
    node_rank = node_comm.Get_rank()
    
    # Calculate run and particle indices
    run_idx = rank // num_particles
    particle_idx = rank % num_particles
    
    if rank == 0:
        print(f"Initialization complete. Starting optimization with {runs} runs and {num_particles} particles per run")

            
    #np.random.seed(rank)  # Ensure different random numbers across processes
    r = np.random.uniform(0, 2 * np.pi)
    particle_position = np.random.uniform(r, r + 2 * np.pi, size=(dimension,))
    particle_velocity = np.random.uniform(-np.pi / 2, np.pi / 2, size=(dimension,))
    
    # Initialize personal best
    personal_best_position = particle_position.copy()
    personal_best_value = cost_fn(particle_position)
    
    # Create run-specific communicator
    comm_run = comm.Split(run_idx, rank)
    # Initialize global best for the run
    all_personal_best_values = comm_run.allgather(personal_best_value)
    all_personal_best_positions = comm_run.allgather(personal_best_position)
    
    global_best_idx = np.argmin(all_personal_best_values)
    global_best_value = all_personal_best_values[global_best_idx]
    global_best_position = all_personal_best_positions[global_best_idx]
    
    # Calculate constants
    omega = 1
    lamb = hp[3]
    tm = hp[2]

    delta = np.abs(personal_best_position - global_best_position)
    mask = delta > np.pi
    attractor = np.where(
        mask,
        np.mod(
            ((hp[0] * personal_best_position + hp[1] * global_best_position) / (hp[0] + hp[1])) + np.pi - r,
            2 * np.pi,
        ) + r,
        (hp[0] * personal_best_position + hp[1] * global_best_position) / (hp[0] + hp[1]),
    )

        # Update amplitude 'A' and angle 'theta'
    A = np.sqrt(
        (particle_position - attractor) ** 2 +
        (1 / omega) ** 2 * (particle_velocity + lamb * (particle_position - attractor)) ** 2
    )
    amp_dis = (delta % (2 * np.pi))
    amp_dis = (np.minimum(2 * np.pi - amp_dis, amp_dis) / 2) * max_cut
    A = np.maximum(A, amp_dis)
    theta = np.arccos((particle_position - attractor) / A)

    #Initialize time
    t = np.zeros(dimension)
    
    # Initialize storage for velocity magnitudes and global best values
    velocity_magnitudes = np.zeros(max_iterations)
    gb = []

    dead = False

    for iteration in range(max_iterations):
        if not dead:
            t += np.random.rand(dimension) * tm
            A = A * np.exp(-lamb * t)
            delta1 = np.abs(personal_best_position - global_best_position) % (2 * np.pi)
            a_dist = (np.minimum(2 * np.pi - delta1, delta1) / 2) * max_cut
            A = np.maximum(A, a_dist)
            particle_position = (A * np.cos(omega * t + theta)) + attractor
            particle_velocity = A * (-omega * np.sin(omega * t + theta) - lamb * np.cos(omega * t + theta))
            # Evaluate the cost function
            current_value = cost_fn(particle_position)
            # Update personal best if necessary
            if current_value < personal_best_value:
                personal_best_value = current_value
                personal_best_position = particle_position.copy()
                personal_best_position = np.mod(personal_best_position-r,2*np.pi)+r
                t = np.zeros(dimension)
            # Update attractor
            delta = np.abs(personal_best_position - global_best_position)
            mask = delta > np.pi
            attractor = np.where(mask,np.mod(((hp[0] * personal_best_position + hp[1] * global_best_position) / (hp[0] + hp[1])) + np.pi - r,2 * np.pi) + r,(hp[0] * personal_best_position + hp[1] * global_best_position) / (hp[0] + hp[1]),)
            
            # Recalculate amplitude 'A' and angle 'theta'
            A1 = np.sqrt((particle_position - attractor) ** 2 +(1 / omega) ** 2 * (particle_velocity + lamb * (particle_position - attractor)) ** 2)
            a_dist = (delta % (2 * np.pi))
            a_dist = (np.minimum(2 * np.pi - a_dist, a_dist) / 2) * max_cut
            A = np.maximum(np.maximum(A, A1), a_dist)
            cos_theta = (particle_position - attractor) / A
            if np.any(cos_theta < -1) or np.any(cos_theta > 1) or np.isnan(cos_theta).any():
               dead = True
               personal_best_value = np.inf  
               print(f"Particle {rank} has been killed due to invalid theta at iteration {iteration}.")# Set to infinity to exclude from global best
            else:
               theta = np.arccos(cos_theta)
                   
        else:
            # Particle is dead; skip updates
            pass
        # Gather personal bests to update the global best within the run
        all_personal_best_values = comm_run.allgather(personal_best_value)
        all_personal_best_positions = comm_run.allgather(personal_best_position)
    
        # Determine the new global best
        global_best_idx = np.argmin(all_personal_best_values)
        new_global_best_value = all_personal_best_values[global_best_idx]
        new_global_best_position = all_personal_best_positions[global_best_idx]
                
        # Update global best if improved
        if new_global_best_value < global_best_value:
            global_best_value = new_global_best_value
            global_best_position = new_global_best_position.copy()
            t = np.zeros(dimension)  # Reset time 't' when global best is updated

            # Update attractor with the new global best
            delta = np.abs(personal_best_position - global_best_position)
            mask = delta > np.pi
            attractor = np.where(
                mask,
                np.mod(
                    ((hp[0] * personal_best_position + hp[1] * global_best_position) / (hp[0] + hp[1])) + np.pi - r,
                    2 * np.pi,
                ) + r,
                (hp[0] * personal_best_position + hp[1] * global_best_position) / (hp[0] + hp[1]),
            )
    
            # Recalculate amplitude 'A' and angle 'theta' with new global best
            A1 = np.sqrt(
                (particle_position - attractor) ** 2 +
                (1 / omega) ** 2 * (particle_velocity + lamb * (particle_position - attractor)) ** 2
            )
            a_dist = (delta % (2 * np.pi))
            a_dist = (np.minimum(2 * np.pi - a_dist, a_dist) / 2) * max_cut
            A = np.maximum(np.maximum(A, A1), a_dist)
            cos_theta = (particle_position - attractor) / A
            if np.any(cos_theta < -1) or np.any(cos_theta > 1) or np.isnan(cos_theta).any():
               dead = True
               personal_best_value = np.inf  # Set to infinity to exclude from global best
               print(f"Particle {rank} has been killed due to invalid theta at iteration {iteration}.")
            else:
               theta = np.arccos(cos_theta)
            theta = np.arccos((particle_position - attractor) / A)
        
                # Record the magnitude of the particle's velocity
        velocity_magnitudes[iteration] = np.linalg.norm(particle_velocity) if not dead else 0
    
        # Record the global best value at this iteration
        gb.append(global_best_value)
        if rank == 0:
            if 'pbv_history' not in locals():
                pbv_history = []
            pbv_history.append(all_personal_best_values.copy())
        

    # Local results from this particle
    e_min_local = personal_best_value  # The best cost found by this particle
    vectors_local = personal_best_position  # The best position found by this particle
    vel_mag_local = velocity_magnitudes  # Velocity magnitudes over iterations
    gb_local = gb  # Global best values over iterations
    
    # Current end of your hopso function:
    # Gather results from all particles
    all_e_min_run = comm_run.gather(e_min_local, root=0)
    all_vectors_run = comm_run.gather(vectors_local, root=0)
    all_vel_mag_run = comm_run.gather(vel_mag_local, root=0)
    all_gb_run = comm_run.gather(gb_local, root=0)
    
    if comm_run.rank == 0:  # Only the root process of each run saves data
        save_to_hdf5(comm, rank, run_idx, all_e_min_run, all_vectors_run, all_vel_mag_run, all_gb_run)
    