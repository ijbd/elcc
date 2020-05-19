import sys

from elcc_impl import main

# Parameters

simulation = dict()
files = dict()
system = dict()
generator = dict()

########## Generic ##########

simulation["year"] = 2018
simulation["iterations"] = 1000 # number of iterations for monte carlo simulation
simulation["rm generators iterations"] = 100 # number of iterations used for removing generators (smaller to save time)
simulation["target lolh"] = 2.4 # loss-of-load-hours per year (2.4 is standard)
simulation["shift load"] = 0 # +/- hours
simulation["debug"] = True # print all information flagged for debug

######## files ########

files["demand file"] = "../demand/PACE.csv"
files["eia folder"] = "../eia8602018/"
files["solar cf file"] = "../wecc_powGen/"+str(simulation["year"])+"_solar_ac_generation.nc"
files["wind cf file"] = "../wecc_powGen/"+str(simulation["year"])+"_wind_ac_generation.nc"

########## System ###########

system["setting"] = "none" # none, save, or load
system["region"] = "PACE" # identify the nerc region or balancing authority (e.g. "PACE", "WECC", etc.)
system["conventional efor"] = .07
system["RE efor"] = 1.0 #set to 1 to remove all W&S generators from current fleet
system["derate conventional"] = False #decrease conventional generators' capacity by 5%
system["oldest year"] = 0 #remove conventional generators older than this year

######## Generator ##########

generator["type"] = "solar" #solar or wind 
generator["nameplate"] = 100 #MW
generator["lat"] = 41
generator["lon"] = -112
generator["efor"] = 0 #0.05 originally

JOB = int(sys.argv[1])


if JOB == 1:
    # default pace
    main(simulation,files,system,generator)


if JOB == 2:
    files["demand file"] = "../demand/WESTERN_IC.csv"
    system["region"] = "WECC"
    # default wecc
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
