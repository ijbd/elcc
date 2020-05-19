#!/bin/bash

num_jobs=1

for job in $(seq 1 $num_jobs)
do
    echo "Running Job: $job"
    echo sbatch elcc_batch_job.sbat $job
done

