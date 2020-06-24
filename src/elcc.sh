#!/bin/bash

root_directory="testing/"

####################### IMPLEMENTATION #######################

    run_job() {
        parameter_string=''
        for key in "${!parameters[@]}"
        do
            parameter_string="$parameter_string $key ${parameters[$key]}"
        done
        sbatch elcc_batch_job.sbat $root_directory $parameter_string
    }

    # parameters for job
    declare -A parameters 

########################### JOBS #############################

    # control
    echo "Running: No storage"
    run_job

    # experiments
    parameters["supplemental_storage"]="True"
    
    for storage_power_capacity in 3000
    do
        parameters["supplemental_storage_power_capacity"]=$storage_power_capacity
        for storage_duration in {1..10}
            do
            storage_energy_capacity=$(($storage_power_capacity * $storage_duration))
            parameters["supplemental_storage_energy_capacity"]=$storage_energy_capacity

            echo "Running: " $storage_power_capacity "MW w/" $storage_energy_capacity "MWh"
            run_job
            done
    done


