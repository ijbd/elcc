import sys

from elcc_impl import main

# Parameters

simulation = dict()
files = dict()
system = dict()
generator = dict()

####################################### DEFAULT #############################################

########## Generic ##########

simulation["year"] = 2018
simulation["region"] = "PACE" # identify the nerc region or balancing authority (e.g. "PACE", "WECC", etc.)
simulation["iterations"] = 1000 # number of iterations for monte carlo simulation
simulation["rm generators iterations"] = 100 # number of iterations used for removing generators (smaller to save time)
simulation["target reliability"] = 2.4 # loss-of-load-hours per year (2.4 is standard)
simulation["shift load"] = 0 # +/- hours
simulation["debug"] = True # print all information flagged for debug

######## files ########

files["output directory"] = "./"
files["eia folder"] = "../eia8602018/"
files["benchmark FORs file"] =  "../efor/Temperature_dependent_for_realtionships.xlsx"
files["total interchange folder"] = "../total_interchange/"
########## System ########### 

# Adjust parameters of existing fleet
system["setting"] = "none" # none, save, or load
system["derate conventional"] = False #decrease conventional generators' capacity by 5%
system["oldest year"] = 0 #remove conventional generators older than this year

######### Outages ###########

system["conventional efor"] = .05 #ignored if temperature-dependent FOR is true
system["renewable efor"] = .05 #set to 1 to ignore all W&S generators from current fleet
system["temperature dependent FOR"] = True #implemnts temeprature dependent forced outage rates for 6 known technologies
system["temperature dependent FOR indpendent of size"] = True #implemnts temperature dependent forced outage rates for all generators, 
                                                            #if false only applies to generators greater then 15 MW, ignore if not using temp dependent FORs
system["enable total interchange"] = True #gathers combined imports/exports data for balancing authority N/A for WECC

######### Storage ###########

system["dispatch strategy"] = "reliability" # "reliability" or "arbitrage"
system["storage efficiency"] = .8 #roundtrip 
system["storage efor"] = 0
system["fleet storage"] = True #include existing fleet storage 
system["supplemental storage"] = False # add supplemental storage to simulate higher storage penetration
system["supplemental storage power capacity"] = 100 # MW
system["supplemental storage energy capacity"] = 100 # MWh

######## Generator ##########

generator["generator type"] = "solar" #solar or wind 
generator["nameplate"] = 100 #MW
generator["lat"] = 41
generator["lon"] = -112
generator["efor"] = .05 

###### Added Storage ########

generator["generator storage"] = False #find elcc of additional storage
generator["generator storage power capacity"] = 100 #MW
generator["generator storage energy capacity"] = 100 #MWh 

##############################################################################################

# handle arguments depending on job based on key-value entries. for multi-word keys, use underscores.
#
#   e.g.        python elcc_master.py year 2017 region WECC print_debug False 
#
parameters = sys.argv[1:]

i = 0
while i < len(parameters):
    key = parameters[i].replace('_',' ')
    value = parameters[i+1]
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
    i += 2 

# dependent parameters

files["demand file"] = "../demand/"+simulation["region"]+".csv"
files["solar cf file"] = "../wecc_powGen/"+str(simulation["year"])+"_solar_generation_cf.nc"
files["wind cf file"] = "../wecc_powGen/"+str(simulation["year"])+"_wind_generation_cf.nc"
files["temperature file"] = "../efor/temperatureDataset"+str(simulation["year"])+".nc"

# save time for stupidity

if files["output directory"][-1] != '/': files["output directory"] += '/'

# run program
main(simulation,files,system,generator)
