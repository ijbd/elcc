import sys, os
from os import path

folder_name = "elcc."

folder_type = sys.argv[1] 

if folder_type == "parameters":
    # add each passed parameter
    for parameter in sys.argv[2:]:
        folder_name += parameter + "__"

if folder_type == "manual":
    folder_name += sys.argv[2]

# add tag
folder_name += ".out"
folder_name.replace('/','.')

# Error Handling
if path.exists(folder_name):
    print("Duplicate folder encountered:",folder_name)
    print("failed/")
    quit()

# Create directory
os.system("mkdir "+folder_name)

# Return directory
print(folder_name + "/")