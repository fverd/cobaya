#!/usr/bin/env bash
#
#SBATCH --job-name=testimpl

#SBATCH --mail-type=END
#SBATCH --mail-user=fverdian@sissa.it
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=70
#SBATCH --mem=46G

#SBATCH --time=48:00:00  
#
#SBATCH --partition=batch
#SBATCH --output=Slurm-output/%x.o%j
export OMP_NUM_THREADS=$((${SLURM_CPUS_PER_TASK}/2))
export RDMAV_FORK_SAFE=1

cobaya-run FRA-params/pbj_boss_b1only.yaml -f