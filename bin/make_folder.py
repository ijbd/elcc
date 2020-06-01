import sys, os
from os import path

folder_name = ""

# add each passed parameter
for parameter in sys.argv[1:]:
    folder_name += parameter + "_"

# add tag
folder_name += ".elcc"
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