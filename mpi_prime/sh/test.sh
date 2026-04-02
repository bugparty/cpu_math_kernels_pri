#!/bin/bash
pushd sh
pwd
output_log="sieve2_5.o"
rm $output_log
# Submit job and store output in variable
#output=$(sbatch sh/sieve2_5.sh)

# Use awk to extract the job ID
job_id=$(sbatch sieve2_5.sh | awk '{print $4}')

# Print the job ID
echo "Task id is: $job_id , wait for finish"
# Check job status every second until the job ID no longer appears
while true; do
    # Check if the job ID exists in squeue output
    if ! squeue -u $USER | grep -q "$job_id"; then
        echo "job $job_id finished"
        break
    fi
    # Check again every second
    sleep 1
done
cat ./$output_log
popd


