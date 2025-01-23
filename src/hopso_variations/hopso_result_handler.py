import os
import h5py
from pathlib import Path

def get_slurm_job_id():
    """
    Get the SLURM job ID from environment variables.
    Returns 'local' if not running under SLURM.
    """
    return os.environ.get('SLURM_JOB_ID', 'local')

def save_to_hdf5(comm, rank, run_idx, all_e_min_run, all_vectors_run, all_vel_mag_run, all_gb_run, save_dir="results"):
    """
    Save HOPSO results to HDF5 with SLURM job ID in filename.
    Only the root process of each run saves data.
    
    Parameters:
    -----------
    comm : MPI.Comm
        MPI communicator
    rank : int
        Process rank
    run_idx : int
        Run index
    all_e_min_run : array-like
        Minimum energies for all particles in the run
    all_vectors_run : array-like
        Best positions for all particles in the run
    all_vel_mag_run : array-like
        Velocity magnitudes for all particles in the run
    all_gb_run : array-like
        Global best values over iterations
    save_dir : str, optional
        Directory to save results (default: "results")
    """
    # Get SLURM job ID
    job_id = get_slurm_job_id()
    
    # Create results directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Create HDF5 file for this run with SLURM job ID in the filename
    filename = f"{save_dir}/hopso_run_{run_idx}_slurm_{job_id}.h5"
    
    with h5py.File(filename, 'w') as f:
        # Save the arrays directly
        f.create_dataset('minimum_energies', data=all_e_min_run)
        f.create_dataset('best_positions', data=all_vectors_run)
        f.create_dataset('velocity_magnitudes', data=all_vel_mag_run)
        f.create_dataset('global_best', data=all_gb_run[0])
        
        # Add job metadata
        f.attrs['slurm_job_id'] = job_id
        f.attrs['run_index'] = run_idx
        
        # Add additional SLURM metadata if available
        slurm_metadata = {
            'SLURM_JOB_NAME': os.environ.get('SLURM_JOB_NAME', 'N/A'),
            'SLURM_SUBMIT_DIR': os.environ.get('SLURM_SUBMIT_DIR', 'N/A'),
            'SLURM_JOB_NODELIST': os.environ.get('SLURM_JOB_NODELIST', 'N/A'),
            'SLURM_NTASKS': os.environ.get('SLURM_NTASKS', 'N/A'),
            'SLURM_JOB_CPUS_PER_NODE': os.environ.get('SLURM_JOB_CPUS_PER_NODE', 'N/A')
        }
        
        # Save SLURM metadata as attributes
        for key, value in slurm_metadata.items():
            f.attrs[key] = value