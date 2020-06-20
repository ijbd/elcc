#!/bin/bash

#1
fleet_storage="False"
supplemental_storage="False"
generator_storage="False"
storage_for=".05"

output_folder=$(python make_folder.py manual no_storage_benchmark)

echo "sbatch elcc_batch_job.sbat" $output_folder $fleet_storage $supplemental_storage $generator_storage $storage_for

#2
fleet_storage="True"
supplemental_storage="True"
generator_storage="True"
storage_for=".05"

output_folder=$(python make_folder.py manual fleet_storage__generator_storage__.05_generator_efor__)

echo "sbatch elcc_batch_job.sbat" $output_folder $fleet_storage $supplemental_storage $generator_storage $storage_for

#2
fleet_storage="True"
supplemental_storage="True"
generator_storage="True"
storage_for=".05"

output_folder=$(python make_folder.py manual fleet_storage__generator_storage__.05_generator_efor__)

echo "sbatch elcc_batch_job.sbat" $output_folder $fleet_storage $supplemental_storage $generator_storage $storage_for

