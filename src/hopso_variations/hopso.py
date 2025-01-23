


# Implementing HOPSO:
# ~~~~~~~~~~~~~~~~~~~~~~~



# HOPSO algorithm

import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from costF_2q_IvaH2_qiskit import objective_function_1
from time import perf_counter

def hopso(cost_fn, hp, num_particles, runs, dimension,max_cut,e_min, vectors, velocities, vel_mag, gbest, max_iterations, file_name):
    # Calculate constants

    s = hp[3]
    lamb = s * (((max_iterations*num_particles)/num_particles) ** -1)

    omega = 1
    tm = hp[2]


    for _ in range(runs):
        # INITIALIZE PARTICLES
        r = np.random.uniform(0,2*np.pi)
        particles_position = np.array(np.random.uniform(r, r+2*np.pi, size=(num_particles, dimension)))
        particles_velocity = np.array(np.random.uniform(-np.pi/2, np.pi/2, size=(num_particles, dimension)))

        # INITIALIZE PERSONAL BEST POSITION
        personal_best_positions = particles_position.copy()

        # INITIALIZE VELOCITY MAGNITUDES
        velocity_magnitudes = np.zeros((max_iterations, num_particles))

        # EVALUATE THE FUNCTION AT INITIAL POSITIONS
        personal_best_values = np.array([cost_fn(p) for p in personal_best_positions])
        global_best_index = np.argmin(personal_best_values)
        global_best_energy = cost_fn(personal_best_positions[global_best_index])
        global_best_position = personal_best_positions[global_best_index].copy()
        gb_position = np.zeros((num_particles, dimension))
        gb_position[:] = global_best_position
        gb = []

        # Initialize iteration counter
        iteration = 0
        # CALCULATE INITIAL ATTRACTORS
        if np.any(np.absolute(personal_best_positions-gb_position) > np.pi):
           mask = np.absolute(personal_best_positions-gb_position) > np.pi
           attractors = np.where(mask,
                                 np.mod(((hp[0] * personal_best_positions + hp[1] * gb_position) / (hp[0] + hp[1])) + np.pi-r,2*np.pi)+r,
                                 ((hp[0] * personal_best_positions + hp[1] * gb_position) / (hp[0] + hp[1])))
        else:
            attractors = (hp[0] * personal_best_positions + hp[1] * gb_position) / (hp[0] + hp[1])
        # attractors = (hp[0] * personal_best_positions + hp[1] * gb_position) / (hp[0] + hp[1])


        A = np.sqrt((particles_position-attractors)**2+(1/(omega))**2*(particles_velocity+(lamb)*(particles_position-attractors))**2)
        amp_dis = (np.abs(personal_best_positions - gb_position)) % (2*np.pi)
        amp_dis = (np.minimum(2*np.pi-amp_dis,amp_dis)/2)*max_cut
        A = np.maximum(A, amp_dis)
        theta = np.arccos((particles_position-attractors)/A)

        # Update particle velocities and positions
        x_list = []
        v_list = []
        A1 = [[] for _ in range(num_particles)]
        delta1 = [[] for _ in range(num_particles)]

        t = np.zeros((num_particles,dimension))

        while iteration < max_iterations:
            t += np.random.rand(num_particles,dimension)*tm
            A = A* np.exp(-lamb * t)
            delta = (np.abs(personal_best_positions - gb_position)) % (2*np.pi)
            a_dist = (np.minimum(2*np.pi-delta,delta)/2)*max_cut
            A = np.maximum(A,a_dist)
            particles_position = (A * np.cos(omega * t  + theta)) + attractors
            x_list.append(particles_position )
            particles_velocity = A * (-omega * np.sin(omega *t  + theta) - lamb * np.cos(omega * t + theta))
            v_list.append(particles_velocity)


            # Update personal best positions
            for i in range(num_particles):
                current_value = cost_fn(particles_position[i])
                if current_value < personal_best_values[i]:
                    personal_best_values[i] = current_value
                    personal_best_positions[i] = np.mod(personal_best_positions[i]-r,2*np.pi)+r
                    t[i] = 0  # Reset t for the updated particle
                    if np.any(np.absolute(personal_best_positions[i]-gb_position[i]) > np.pi):
                      # Update attractor position dimension-wise
                      mask = np.absolute(personal_best_positions[i]-gb_position[i]) > np.pi
                      attractors[i] = np.where(mask,
                                            np.mod(((hp[0] * personal_best_positions[i] + hp[1] * gb_position[i]) / (hp[0] + hp[1])) + np.pi-r,2*np.pi)+r,
                                            (hp[0] * personal_best_positions[i] + hp[1] * gb_position[i]) / (hp[0] + hp[1]))
                    else:
                        # No dimension-wise update needed, standard update
                        attractors[i] = (hp[0] * personal_best_positions[i] + hp[1] * gb_position[i]) / (hp[0] + hp[1])

                    delta1[i] = (np.abs(personal_best_positions[i] - gb_position[i])) % (2*np.pi)
                    delta1[i] = (np.minimum(2*np.pi-delta1[i],delta1[i])/2)*max_cut
                    A1[i] = np.sqrt((particles_position[i] - attractors[i]) ** 2 + (1/(omega))** 2 * (particles_velocity[i] + lamb* (particles_position[i] - attractors[i])) ** 2)
                    A[i] = np.maximum(np.maximum(A[i],A1[i]),delta1[i])
                    theta[i] = np.arccos((particles_position[i] - attractors[i]) / A[i])

            if np.min(personal_best_values) < global_best_energy:
                global_best_index = np.argmin(personal_best_values)
                global_best_position = personal_best_positions[global_best_index].copy()
                global_best_energy = personal_best_values[global_best_index]
                gb_position[:] = global_best_position
                t[:] = 0
                if np.any(np.absolute(personal_best_positions-gb_position) > np.pi):
                   mask = np.absolute(personal_best_positions-gb_position) > np.pi
                   attractors = np.where(mask,
                                         np.mod(((hp[0] * personal_best_positions + hp[1] * gb_position) / (hp[0] + hp[1])) + np.pi-r,2*np.pi)+r,
                                         (hp[0] * personal_best_positions + hp[1] * gb_position) / (hp[0] + hp[1]))
                else:
                    attractors = (hp[0] * personal_best_positions + hp[1] * gb_position) / (hp[0] + hp[1])
                delta_all = (np.abs(personal_best_positions - gb_position)) % (2*np.pi)
                amp_dist_all = (np.minimum(2*np.pi-delta_all,delta_all)/2)*max_cut
                A1 = np.sqrt((particles_position - attractors) ** 2 + (1/(omega))** 2 * (particles_velocity + lamb * (particles_position - attractors)) ** 2)
                A = np.maximum(np.maximum(A,A1),amp_dist_all)
                theta = np.arccos((particles_position - attractors) / A)

            velocity_magnitudes[iteration, :] = np.linalg.norm(particles_velocity, axis=1)
            gb.append(cost_fn(global_best_position))

            # # Increment iteration counter
            iteration += 1


        gbest.append(gb)
        e_min.append(np.float64(cost_fn(global_best_position)))
        vectors.append(global_best_position)
        vel_mag.append(velocity_magnitudes)

        print(e_min)

    # Save global best per run to a file
    #with open(file_name, 'wb') as file:
    #    pickle.dump(e_min, file)




# ======== Run HOPSO =======
# User Input Parameters: HOPSO
  # HOPSO/PSO hyperparameters--- c1, c2, tm, s (NOT LAMBDA) 
# s=1, with 500-iter, 10p means ==B = 5000 => (5K/10P)-1 = 500**-1

hp = [1, 1, 2*np.pi, 10]
maxcut = 4.10             # A_threshold
num_particles = 10          # Number of particles
max_iterations = 500



# ====== Data Collection =======
# Data Collection Details
e_min = []
vectors = []
velocities = []
vel_mag = []
gbest = []
start_time = perf_counter()
hopso(objective_function_1, hp, num_particles, 1, 6, maxcut, e_min, vectors, velocities, vel_mag, gbest, max_iterations, "hopso_result")
end_time = perf_counter()
print(str(end_time - start_time) + "s")