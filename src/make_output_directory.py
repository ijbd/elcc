import sys, os
from os import path

root_directory = sys.argv[1]

output_directory = root_directory+"elcc.__"

# default parameters
if len(sys.argv) == 2:
    output_directory += "default.out"

# handle all parameters
else: 
    # add each passed parameter
    for parameter in sys.argv[1:]:
        if parameter.find('/') == -1: #don't include files/directories
            output_directory += parameter + "__"

    # add tag
    output_directory += ".out"
    output_directory.replace('/','.')

# Error Handling
if path.exists(output_directory):
    print("Duplicate folder encountered:",output_directory)
    print("failed/")
    quit()

# Create directory
os.system("mkdir "+output_directory)

# Return directory
print(output_directory+"/")