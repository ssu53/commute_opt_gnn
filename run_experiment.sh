#!/bin/bash
#BATCH -J EXP
#SBATCH -A COMPUTERLAB-SL2-CPU
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16000
#SBATCH --partition=icelake
#SBATCH --mail-type=FAIL
#SBATCH --output=/home/is473/rds/hpc-work/commute_opt_gnn/slurm/%j.out
#! ############################################################
# . /etc/profile.d/modules.sh                # Leave this line (enables the module command)
# module purge                               # Removes all modules still loaded
# module load rhel8/default-amp
# module load cuda/11.1 intel/mkl/2017.4

source /home/is473/miniconda3/etc/profile.d/conda.sh
conda activate /home/is473/rds/hpc-work/commute_opt_gnn/l65-env

## YOUR SCRIPT DOWN HERE
JOBID=$SLURM_JOB_ID
LOG=/home/is473/rds/hpc-work/commute_opt_gnn/slurm/log.$JOBID
ERR=/home/is473/rds/hpc-work/commute_opt_gnn/slurm/err.$JOBID

echo -e "JobID: $JOBID\n======" > $LOG
echo "Time: `date`" >> $LOG
echo "Running on master node: `hostname`" >> $LOG

python train.py --config_fn "ColourInteract-OOD.yaml" --c2_over_c1 10.0 >> $LOG 2> $ERR
echo "Time: `date`" >> $LOG