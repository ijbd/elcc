import sys
import os
import numpy as np
import pandas as pd

from elcc_impl import main


# Parameters

simulation = dict()
files = dict()
system = dict()
generator = dict()

####################################### DEFAULT #############################################

########## Generic ##########

simulation["year"] = 2018
simulation["region"] = ["PACE"] # identify the nerc region or balancing authority (e.g. "PACE", "WECC", etc.)
simulation["iterations"] = 10000 # number of iterations for monte carlo simulation
simulation["target reliability"] = 2.4 # loss-of-load-hours per year (2.4 is standard)
simulation["shift load"] = 0 # +/- hours
simulation["debug"] = False # print all information flagged for debug

######## files ########

files["root directory"] = None # change to valid directory to create output directory
files["output directory"] = './'
files["eia folder"] = "../eia8602018/"
files["benchmark FORs file"] =  "../efor/Temperature_dependent_for_realtionships.xlsx"
files["total interchange folder"] = "../total_interchange/"
files["saved systems folder"] = "/scratch/mtcraig_root/mtcraig1/shared_data/elccJobs/savedSystems/"

########## System ########### 

# Adjust parameters of existing fleet
system["system setting"] = "save" # none or save (save will load existing fleet capacity or save new folder)
system["oldest year"] = 0 #remove conventional generators older than this year
system["renewable multiplier"] = 1 #multiplier for existing renewables

######### Outages ###########

system["conventional efor"] = .05 #ignored if temperature-dependent FOR is true
system["renewable efor"] = .05 #set to 1 to ignore all W&S generators from current fleet
system["temperature dependent FOR"] = True #implemnts temeprature dependent forced outage rates for 6 known technologies
system["temperature dependent FOR indpendent of size"] = True #implemnts temperature dependent forced outage rates for all generators, 
                                                            #if false only applies to generators greater then 15 MW, ignore if not using temp dependent FORs
system["enable total interchange"] = True #gathers combined imports/exports data for balancing authority N/A for WECC

######### Storage ###########

system["dispatch strategy"] = "reliability" 
system["storage efficiency"] = .8 #roundtrip 
system["storage efor"] = 0
system["supplemental storage"] = False # add supplemental storage to simulate higher storage penetration
system["supplemental storage power capacity"] = 1000 # MW
system["supplemental storage energy capacity"] = 1000 # MWh

######## Generator ##########

generator["generator type"] = "solar" #solar or wind 
generator["nameplate"] = 1000 #MW
generator["latitude"] = 41
generator["longitude"] = -112
generator["efor"] = .05

###### Added Storage ########

generator["generator storage"] = False #find elcc of additional storage
generator["generator storage power capacity"] = 500 #MW
generator["generator storage energy capacity"] = 500 #MWh 

##############################################################################################


# handle arguments depending on job based on key-value entries. for multi-word keys, use underscores.
#
#   e.g.        python elcc_master.py year 2017 region WECC print_debug False 

parameters = dict()

params = sys.argv[1:]

# fill dictionary with arguments
i = 0
while i < len(params):

    key = params[i].replace('_',' ')
    value = params[i+1]
    parameters[key] = value

    i += 2 


# fill job parameter dictionaries with passed arguments 
for key in parameters:

    value = parameters[key]

    if key == 'region':
        simulation['region'] = parameters['region'].split()

    else:
        for param_set in [simulation, files, system, generator]:
            if key in param_set:

                param_set[key] = str(value)

                # handle numerical arguments
                try:
                    # floats
                    float_value = float(value)
                    param_set[key] = float_value
        
                    # ints
                    if float(value) == int(value):
                        param_set[key] = int(value)
                except:
                    pass

                # handle boolean arguments
                if value == "True": param_set[key] = True
                elif value == "False": param_set[key] = False

            

# fix dependent parameters
files["solar cf file"] = "/scratch/mtcraig_root/mtcraig1/shared_data/merraData/cfs/wecc/"+str(simulation["year"])+"_solar_generation_cf.nc"
files["wind cf file"] = "/scratch/mtcraig_root/mtcraig1/shared_data/merraData/cfs/wecc/"+str(simulation["year"])+"_wind_generation_cf.nc"
files["temperature file"] = "/scratch/mtcraig_root/mtcraig1/shared_data/merraData/resource/wecc/processed/cordDataWestCoastYear"+str(simulation["year"])+".nc"

# handle output directory and print location
root_directory = files["root directory"]

redirect_output = not root_directory is None

if redirect_output:

    if root_directory[-1] != '/': root_directory += '/'
    if not os.path.exists(root_directory):
        print('Invalid directory:', root_directory)
        sys.exit(1)

    output_directory = root_directory+"elcc.__"

    # add each passed parameter
    for key in sorted(parameters):
        if parameters[key].find('/') == -1 and not key in ["root directory", "output directory"]: #don't include files/directories
            output_directory += key.replace(' ','_') + '__' + parameters[key].replace(' ','_') + '__'

    # add tag
    output_directory += ".out"
    output_directory.replace('\"','')

    # Error Handling
    if os.path.exists(output_directory):
        print("Duplicate folder encountered:",output_directory)
        sys.exit(1)

    # Create directory
    os.system("mkdir "+output_directory)

    output_directory += '/'

    sys.stdout = open(output_directory + 'print.out', 'w')
    sys.stderr = sys.stdout

    files['output directory'] = output_directory

# time savers
if simulation["region"] == ["WECC"]:
    system['enable total interchange'] = False
    system['oldest year'] = 1975

if files["output directory"][-1] != '/': files["output directory"] += '/'
if files["saved systems folder"][-1] != '/': files["saved systems folder"] += '/'

# TEPPC regions

TEPPC_regions = pd.read_csv('../demand/_regions.csv').fillna('nan')

regions = np.append([region for region in simulation["region"] if not region in(TEPPC_regions.columns)],
                    [TEPPC_regions[region].values.flatten().astype(str) for region in simulation["region"] if region in TEPPC_regions.columns]).flatten()
regions = np.unique(regions[regions != 'nan'])

if len(regions) == 0:
    print("Invalid region(s):",simulation["region"])
    sys.exit(1)

simulation["all regions"] = regions

# run program
main(simulation,files,system,generator)
