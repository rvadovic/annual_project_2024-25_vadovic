# Harmonic Oscillator PSO

This project implemets HOPSO algorithm to find the ground-state energy of molecules using Qiskit

## Requirements

- Intel MPI (2021+)  
- Python 3.10+  
- Conda (reccomended)  
- Qiskit, NumPy, mpi4py libraries  

## Set

In [run.py](https://github.com/rvadovic/annual_project_2024-25_vadovic/blob/5caa93b8bdc93f921386f20e30fd8c534ceb296a/src/hopso_variations/run.py) choose any of the imported HOPSO variations and set hyperparameters and other parameters

### Set hyperparameters

hp[0] - c1 - **cognitive coefficient**, attraction towards particle's best known position  
hp[1] - c2 - **social coefficient**, attraction towards swarm's best known position  
hp[2] - tm - **time multiplier**, controls the rate at which time t evolves for each particle  
hp[3] - lambda - **damping coefficient**, controls how fast the amplitude A decays over time  

maxcut - smallest possible amplitude A

## Run

`mpiexec -n (number of particles) python run.py`

