
from ijbd_elcc import main

# Parameters

########## Generic ##########

year=2018
num_iterations=1000
tgtAnnualLOLH = 2.4

######## Directories ########

demand_file="../demand/PACE.csv"
eia_folder="../eia8602018/"
solar_file="../wecc_powGen/2018_solar_ac_generation.nc"
wind_file="../wecc_powGen/2018_wind_ac_generation.nc"

########## System ###########

# system_setting="system-0-saved.npy" #none, save (save processed system), or filename (of saved processed system)
system_setting="none"

balancing_authority="0" #otherwise set 0  or PACE
#or
nerc_region="WECC" # DO NOT identify both a balancing authority and a nerc region

conventional_efor=.07
vg_efor=1 #set to 1 to remove all W&S generators from current fleet
derate_conventional="True"
oldest_year_manual=1950 #set to 0 if not known, used for removing generators

######## Generator ##########

generator_type="solar" #solar or wind
generator_capacity=100 #MW
generator_latitude=41
generator_longitude=-112
generator_efor=0.0 #0.05 originally

for test in [.5,20]:
    print('target lolh:',test)
    main(year,num_iterations,demand_file,eia_folder,solar_file,wind_file,
        system_setting,balancing_authority,nerc_region,conventional_efor,
        vg_efor,derate_conventional,oldest_year_manual,generator_type,
        generator_capacity,generator_latitude,generator_longitude,generator_efor,
        test)    
aa
for dShift in [-1,0,1]:
    print('**************************SHIFT HOURS:',dShift)
    main(year,num_iterations,demand_file,eia_folder,solar_file,wind_file,
        system_setting,balancing_authority,nerc_region,conventional_efor,
        vg_efor,derate_conventional,oldest_year_manual,generator_type,
        generator_capacity,generator_latitude,generator_longitude,generator_efor,
        tgtAnnualLOLH,dShift)

aa

for i in range(4):
    main(year,num_iterations,demand_file,eia_folder,solar_file,wind_file,
        system_setting,balancing_authority,nerc_region,conventional_efor,
        vg_efor,derate_conventional,oldest_year_manual,generator_type,
        generator_capacity,generator_latitude,generator_longitude,generator_efor,
        tgtAnnualLOLH)

print('2017')
for i in range(4):
    main(2017,num_iterations,demand_file,eia_folder,solar_file,wind_file,
        system_setting,balancing_authority,nerc_region,conventional_efor,
        vg_efor,derate_conventional,oldest_year_manual,generator_type,
        generator_capacity,generator_latitude,generator_longitude,generator_efor,
        tgtAnnualLOLH)

# balancing_authority="0" #otherwise set 0   
# #or
# nerc_region="WECC" # DO NOT identify both a balancing authority and a nerc region
# main(year,num_iterations,demand_file,eia_folder,solar_file,wind_file,
#         system_setting,balancing_authority,nerc_region,conventional_efor,
#         vg_efor,derate_conventional,oldest_year_manual,generator_type,
#         generator_capacity,generator_latitude,generator_longitude,generator_efor,
#         tgtAnnualLOLH,dShift)    


for generator_efor in [0.0,.05,.1]:
    print('****************************GENERATOR EFOR:',generator_efor)
    main(year,num_iterations,demand_file,eia_folder,solar_file,wind_file,
        system_setting,balancing_authority,nerc_region,conventional_efor,
        vg_efor,derate_conventional,oldest_year_manual,generator_type,
        generator_capacity,generator_latitude,generator_longitude,generator_efor,
        tgtAnnualLOLH)

generator_efor = 0.05
for conventional_efor in [.025,.075,.125]:
    print('***********************CONVENTIONAL EFOR:',conventional_efor)
    main(year,num_iterations,demand_file,eia_folder,solar_file,wind_file,
        system_setting,balancing_authority,nerc_region,conventional_efor,
        vg_efor,derate_conventional,oldest_year_manual,generator_type,
        generator_capacity,generator_latitude,generator_longitude,generator_efor,
        tgtAnnualLOLH)



