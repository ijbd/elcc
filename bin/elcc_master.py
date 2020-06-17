import sys

from elcc_impl import main
#from elcc_impl_test import main

# Parameters

simulation = dict()
files = dict()
system = dict()
generator = dict()

####################################### DEFAULT #############################################

########## Generic ##########

simulation["year"] = 2018
simulation["region"] = "PACE" # identify the nerc region or balancing authority (e.g. "PACE", "WECC", etc.)
simulation["iterations"] = 500 # number of iterations for monte carlo simulation
simulation["rm generators iterations"] = 100 # number of iterations used for removing generators (smaller to save time)
simulation["target lolh"] = 2.4 # loss-of-load-hours per year (2.4 is standard)
simulation["shift load"] = 0 # +/- hours
simulation["debug"] = True # print all information flagged for debug
simulation["output folder"] = "./"

######## files ########

files["demand file"] = "../demand/"+simulation["region"]+".csv"
files["eia folder"] = "../eia8602018/"
files["solar cf file"] = "../wecc_powGen/"+str(simulation["year"])+"_solar_ac_generation.nc"
files["wind cf file"] = "../wecc_powGen/"+str(simulation["year"])+"_wind_ac_generation.nc"
files["temperature file"] = "../efor/temperatureDataset"+str(simulation["year"])+".nc"
files["benchmark FORs file"] =  "../efor/Temperature_dependent_for_realtionships.xlsx"

########## System ########### 

# Adjust parameters of existing fleet
system["setting"] = "none" # none, save, or load
system["conventional efor"] = .05
system["RE efor"] = 0.0 #set to 1 to remove all W&S generators from current fleet
system["derate conventional"] = False #decrease conventional generators' capacity by 5%
system["oldest year"] = 1950 #remove conventional generators older than this year
system["Temperature-dependent FOR"] = False #implemnts temeprature dependent forced outage rates for 6 known technologies
system["Temperature-dependent FOR indpendent of size"] = True #implemnts temperature dependent forced outage rates for all generators, 
                                                            #if false only applies to generators greater then 15 MW, ignore if not using temp dependent FORs

######### Storage ###########

system["fleet storage"] = True #include existing fleet storage 
system["storage efficiency"] = .8 #roundtrip (as decimal)
system["storage strategy threshold"] = 0 #load threshold for switching between reliability strategy and peak-shaving strategy
system["supplemental storage"] = True # add supplemental storage to simulate higher storage penetration
system["supplemental storage charge rate"] = 100 # MW
system["supplemental storage discharge rate"] = 100 # MW
system["supplemental storage energy capacity"] = 100 # MWh


######## Generator ##########

generator["type"] = "solar" #solar or wind 
generator["nameplate"] = 100 #MW
generator["lat"] = 41
generator["lon"] = -112
generator["efor"] = 1.0 #0.05 originally

###### Added Storage ########

generator["generator storage"] = True #find elcc of additional storage
generator["generator storage charge rate"] = 100 #MW
generator["generator storage discharge rate"] = 100 #MW
generator["generator storage energy capacity"] = 100 #MWh 
generator["generator storage efficiency"] = .8 #roundtrip (as decimal) 

##############################################################################################

# handle arguments depending on job
simulation["output folder"] = 'testing/'

print('*********************** SOLAR ONLY')
system["supplemental storage"] = False
generator["generator storage"] = False

main(simulation,files,system,generator)

print('*********************** SOLAR W/ SUPPLEMENTAL FLEET STORAGE')
system["supplemental storage"] = True
generator["generator storage"] = False

main(simulation,files,system,generator)

print('*********************** SOLAR W/ GENERATOR STORAGE')
system["supplemental storage"] = False
generator["generator storage"] = False

main(simulation,files,system,generator)


###### TESTING ########
TESTING = False

if TESTING:

    for dShift in [-1,0,1]:
        print('**************************SHIFT HOURS:',dShift)

        test_simulation = dict(simulation)
        test_simulation["shift load"] = dShift

        main(test_simulation,files,system,generator)  

    for generator_efor in [0.0,.25,.5]:
        print('****************************GENERATOR EFOR:',generator_efor)

        test_generator = dict(generator)
        test_generator["efor"] = generator_efor

        main(simulation,files,system,test_generator)

    for conventional_efor in [.025,.075,.125]:
        print('***********************CONVENTIONAL EFOR:',conventional_efor)
        
        test_system = dict(system)
        test_system["conventional efor"] = conventional_efor

        main(simulation,files,test_system,test_generator)

    for tgtLolh in [2.4,20]:
        print('************************TARGET LOLH:',tgtLolh)

        test_simulation = dict(simulation) #deep copy
        test_simulation["target lolh"] = tgtLolh

        main(test_simulation,files,system,generator)    

    for i in range(4):
        print('******************************YEAR:,',2017)
        
        test_simulation = dict(simulation)
        test_simulation["year"] = 2017

        main(test_simulation,files,system,generator)
