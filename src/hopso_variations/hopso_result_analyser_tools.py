import h5py
import numpy as np
from pathlib import Path

def load_run(run_idx, job_id=None, save_dir="results"):
    """
    Load data for a specific run.
    
    Returns:
    - final_energy: float
    - global_best_history: array of energies for each iteration
    - final_position: array of final best position
    - velocity_history: array of velocity magnitudes for each iteration
    """
    if job_id is None:
        job_id = os.environ.get('SLURM_JOB_ID', 'local')
        
    filename = f"{save_dir}/hopso_run_{run_idx}_slurm_{job_id}.h5"
    
    with h5py.File(filename, 'r') as f:
        global_best = f['global_best'][:]
        final_energy = global_best[-1]
        final_position = f['best_positions'][:][-1]  # Last position from best_positions
        velocity_history = f['velocity_magnitudes'][:]
        
    return {
        'final_energy': final_energy,
        'global_best_history': global_best,
        'final_position': final_position,
        'velocity_history': velocity_history
    }

def get_all_final_energies(job_id=None, save_dir="results"):
    """
    Get list of final energies from all runs.
    Returns a sorted dictionary {run_idx: final_energy}
    """
    if job_id is None:
        job_id = os.environ.get('SLURM_JOB_ID', 'local')
        
    path = Path(save_dir)
    energies = {}
    
    pattern = f"hopso_run_*_slurm_{job_id}.h5"
    for file_path in path.glob(pattern):
        run_idx = int(file_path.stem.split('_')[2])
        with h5py.File(file_path, 'r') as f:
            final_energy = f['global_best'][:][-1]
            energies[run_idx] = final_energy
            
    return dict(sorted(energies.items()))