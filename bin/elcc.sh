#!/bin/bash

year=2018
for region in "PACE" "WECC"
do
    echo "Running: $year $region"
    sbatch elcc_batch_job.sbat $year $region
    #bash elcc_batch_job.bash $year $region
done

