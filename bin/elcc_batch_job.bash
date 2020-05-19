#!/bin/bash

# make unique folder for this job from parameters passed
output_folder=$(python elcc_make_folder.py "$@")

python -u elcc_master.py "$output_folder" "$@" > "$output_folder"print.out