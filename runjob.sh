#!/usr/bin/env bash
#
#SBATCH --job-name=plot

#SBATCH --mail-type=END
#SBATCH --mail-user=fverdian@sissa.it
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=46G

#SBATCH --time=48:00:00  
#
#SBATCH --partition=batch
#SBATCH --output=Slurm-output/%x.o%j
export OMP_NUM_THREADS=$((${SLURM_CPUS_PER_TASK}/2))
export RDMAV_FORK_SAFE=1

# cobaya-run /home/fverdian/cobaya/FRA-params/AxiCLASS_Q0.yaml -f

python /home/fverdian/cobaya/FRA/cornerplot-ofgrids-script.py