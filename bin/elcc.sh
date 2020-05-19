#!/bin/bash

for job in {1..2}

echo "Running Job: $job"
sbatch elcc_batch_job.sbat $job
