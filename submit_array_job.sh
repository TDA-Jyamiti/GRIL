#!/bin/sh
#SBATCH --job-name=mega_array   # Job name
#SBATCH --nodes=1                   # Use one node
#SBATCH -c 32
#SBATCH --mem-per-cpu=1gb           # Memory per processor
#SBATCH --time=02:00:00             # Time limit hrs:min:sec
#SBATCH --output=array_%A-%a.out    # Standard output and error log
# This is an example script that combines array tasks with
# bash loops to process many short runs. Array jobs are convenient
# for running lots of tasks, but if each task is short, they
# quickly become inefficient, taking more time to schedule than
# they spend doing any work and bogging down the scheduler for
# all users. 
pwd; hostname; date

#Set the number of runs that each SLURM task should do
PER_TASK=100

# Calculate the starting and ending values for this task based
# on the SLURM task and the number of runs per task.
START_NUM=$(( $SLURM_ARRAY_TASK_ID * $PER_TASK))
END_NUM=$(( (($SLURM_ARRAY_TASK_ID + 1)) * $PER_TASK))
echo $START_NUM
echo $END_NUM

# Print the task and run range
echo This is task $SLURM_ARRAY_TASK_ID, which will do runs $START_NUM to $(( $END_NUM - 1 ))

module load anaconda
module load boost
source activate mpml

# Run the loop of runs for this task.
for (( run=$START_NUM; run<END_NUM; run++ )); do
  echo This is SLURM task $SLURM_ARRAY_TASK_ID, run number $run
  #Do your stuff here
  /home/mukher26/.conda/envs/cent7/2020.02-py37/mpml/bin/python main_graph.py --dataset $1 --sample $run
done

date
