import csv
import datetime
import math
import os
import os.path
import sys
from os import path
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from numpy import random
import matplotlib.pyplot as plt

np.random.seed()

year = int(sys.argv[1])
num_iterations = int(sys.argv[2])
demand_file = sys.argv[3]
eia_folder = sys.argv[4]
solar_file = sys.argv[5]
wind_file = sys.argv[6]
system_setting =sys.argv[7]
balancing_authority = sys.argv[8]    
nerc_region = sys.argv[9]
conventional_efor = float(sys.argv[10])
vg_efor = float(sys.argv[11]) 
derate_conventional = bool(sys.argv[12])
oldest_year_manual = int(sys.argv[13]) 
generator_type = sys.argv[14]
generator_capacity = float(sys.argv[15])
generator_latitude = float(sys.argv[16])
generator_longitude = float(sys.argv[17])
generator_efor = float(sys.argv[18])


# Get all necessary information from powGen netCDF files, VRE capacity factos and lat/lons
def get_powGen(solar_file_in, wind_file_in):
    solar = Dataset(solar_file_in)
    wind = Dataset(wind_file_in) #assume solar and wind cover same geographic region

    powGen_lats = np.array(solar.variables['lat'][:])
    powGen_lons = np.array(solar.variables['lon'][:])
    solar_cf = np.array(solar.variables['ac'][:])
    wind_cf = np.array(wind.variables['ac'][:])

    solar.close()
    wind.close()
    return powGen_lats, powGen_lons, solar_cf, wind_cf


#EFFECTS:   Accesses demand data for balancing from local file. Original data acquired from [1].
#
#MODIFIES   returns array containing 8760 hourly load data points for one year. Skips leap day
#
#SOURCE:
#           [1] Tyler Ruggles, & David Farnham. (2019). EIA Cleaned Hourly Electricity Demand Data
#               (Version v1.0_23Oct2019) [Data set]. Zenodo. http://doi.org/10.5281/zenodo.3517197
#
def get_balancing_authority_demand_data(demand_file_in, year_in):
    hourly_peak_load = np.array([])
    datetime_labels = np.genfromtxt(demand_file_in,delimiter=',',skip_header=1,usecols=0,dtype=str)
    all_demand = np.genfromtxt(demand_file_in,delimiter=',',skip_header=1,usecols=4)
    for sample in range(datetime_labels.size):
        if (datetime_labels[sample]).find(str(year_in),0,4) != -1:
            if (datetime_labels[sample]).find(str(year_in)+"0229",0,8) == -1:
                hourly_peak_load = np.append(hourly_peak_load, all_demand[sample])
    return hourly_peak_load


#EFFECTS:   Accesses demand data for region from local file. Original data acquired from [3].
#
#MODIFIES   returns array containing 8760 hourly load data points for one year. Skips leap day
#
#SOURCE:
#           [1] Tyler Ruggles, & David Farnham. (2019). EIA Cleaned Hourly Electricity Demand Data
#               (Version v1.0_23Oct2019) [Data set]. Zenodo. http://doi.org/10.5281/zenodo.3517197
#
def get_regional_demand_data(demand_file_in, year_in):
    hourly_peak_load = np.array([])
    datetime_labels = np.genfromtxt(demand_file_in,delimiter=',',skip_header=1,usecols=0,dtype=str)
    all_demand = np.genfromtxt(demand_file_in,delimiter=',',skip_header=1,usecols=2)
    for sample in range(datetime_labels.size):
        if (datetime_labels[sample]).find(str(year_in),0,4) != -1:
            if (datetime_labels[sample]).find(str(year_in)+"0229",0,8) == -1:
                hourly_peak_load = np.append(hourly_peak_load, all_demand[sample])
    return hourly_peak_load


# Accesses generator capacity and expected forced outage rate.
# All generator data from [2].
#
#           [2] https://www.eia.gov/electricity/data/eia860/
#
#   fleet = [[Capacity(MW)],[EFOR],[YEAR ENTERED OPERATION]]
#
def get_conventional_fleet_old(eia_folder, balancing_authority, nerc_region, year_in, efor_in):
    #sort by balancing authority
    plant_codes = np.array(pd.read_excel(eia_folder+"2___Plant_Y2018.xlsx",header=1,usecols=[2]).values)
    plant_balancing_authorities = np.array(pd.read_excel(eia_folder+"2___Plant_Y2018.xlsx",header=1,usecols=[12],dtype=str).values)
    plant_nerc_region = np.array(pd.read_excel(eia_folder+"2___Plant_Y2018.xlsx",header=1,usecols=[11],dtype=str).values)
    desired_plant_codes = np.array([])

    if str(balancing_authority) != "0":
        for plant in range(plant_balancing_authorities.size):
            if (str(plant_balancing_authorities[plant,0])).find(balancing_authority) != -1:
                desired_plant_codes = np.append(desired_plant_codes, plant_codes[plant])

    if str(nerc_region) != "0":
        for plant in range(plant_balancing_authorities.size):
            if (str(plant_nerc_region[plant,0])).find(nerc_region) != -1:
                desired_plant_codes = np.append(desired_plant_codes, plant_codes[plant])


    generator_plant_codes = np.array(pd.read_excel(eia_folder+"3_1_Generator_Y2018.xlsx",header=1,usecols=[2]).values)
    generator_capacities = np.array(pd.read_excel(eia_folder+"3_1_Generator_Y2018.xlsx",header=1,usecols=[15]).values)
    generator_status = np.array(pd.read_excel(eia_folder+"3_1_Generator_Y2018.xlsx",header=1,usecols=[23]).values)
    generator_year = np.array(pd.read_excel(eia_folder+"3_1_Generator_Y2018.xlsx",header=1,usecols=[26]).values)
    fleet_capacity = np.array([])
    fleet_year = np.array([])
    for generator in range(generator_plant_codes.size):
        if np.any(desired_plant_codes == generator_plant_codes[generator]):
            if (str(generator_status[generator])).find("OP") != -1:
                fleet_capacity = np.append(fleet_capacity, generator_capacities[generator])
                fleet_year = np.append(fleet_year, generator_year[generator])
    #print("Found "+ str(fleet_capacity.size) +" conventional plants")

    fleet_efor = np.ones(fleet_capacity.size) * efor_in
    conventional_system = np.array([fleet_capacity, fleet_efor, fleet_year])
    return conventional_system

def get_conventional_fleet(eia_folder, balancing_authority, nerc_region, year_in, efor_in):
    
    #sort by balancing authority
    plants = pd.read_excel(eia_folder+"2___Plant_Y2018.xlsx",skiprows=1,usecols=["Plant Code","NERC Region","Balancing Authority Code"])
    if str(nerc_region) != "0":
        desired_codes = plants["Plant Code"][plants["NERC Region"] == nerc_region]
    elif str(balancing_authority) != "0":
        desired_codes = plants["Plant Code"][plants["Balancing Authority Code"] == balancing_authority].values
    else:
        print("Fleet Error.")
        return 1

    generators = pd.read_excel(eia_folder+"3_1_Generator_Y2018.xlsx",skiprows=1,\
                                usecols=["Plant Code","Technology","Nameplate Capacity (MW)","Status","Operating Year"],\
                                index_col="Plant Code")
    
    fleet_generators = generators[generators.index.isin(desired_codes)]
    fleet_generators = fleet_generators[fleet_generators["Status"] == "OP"]
    fleet_generators = fleet_generators[fleet_generators["Technology"]!="Solar Photovoltaic"]
    fleet_generators = fleet_generators[fleet_generators["Technology"]!="Onshore Wind Turbine"]
    fleet_generators = fleet_generators[fleet_generators["Technology"]!="Offshore Wind Turbine"]
    
    fleet_capacity = np.array(fleet_generators["Nameplate Capacity (MW)"].values)
    fleet_year = np.array(fleet_generators["Operating Year"].values)

    fleet_efor = np.ones(fleet_capacity.size) * efor_in
    conventional_system = np.array([fleet_capacity, fleet_efor, fleet_year])
    return conventional_system


# Reduce generator capacity by 5% if derate_conventional is True
def derate(derate_conventional, conventional_system):
    if derate_conventional == True:
        conventional_system[0,:] = conventional_system[0,:] * .95
    return conventional_system


# All generator data from [2].
#
#           [2] https://www.eia.gov/electricity/data/eia860/
#
#   vg = [[Nameplate Capacity (MW)],[Latitude],[Longitude],[EFOR]]
#
def get_vg_system(eia_folder, balancing_authority, nerc_region, year_in, efor_in):
    solar_nameplate_capacity = np.array([],dtype=float)
    solar_lat = np.array([],dtype=float)
    solar_lon = np.array([],dtype=float)
    wind_nameplate_capacity = np.array([],dtype=float)
    wind_lat = np.array([],dtype=float)
    wind_lon = np.array([],dtype=float)

    #sort by balancing authority
    plant_codes = np.array(pd.read_excel(eia_folder+"2___Plant_Y2018.xlsx",header=1,usecols=[2]).values)
    plant_balancing_authorities = np.array(pd.read_excel(eia_folder+"2___Plant_Y2018.xlsx",header=1,usecols=[12],dtype=str).values)
    plant_nerc_region = np.array(pd.read_excel(eia_folder+"2___Plant_Y2018.xlsx",header=1,usecols=[11],dtype=str).values)
    lats = np.array(pd.read_excel(eia_folder+"2___Plant_Y2018.xlsx",header=1,usecols=[9]).values)
    lons = np.array(pd.read_excel(eia_folder+"2___Plant_Y2018.xlsx",header=1,usecols=[10]).values)
    desired_plant_codes = np.array([])
    desired_lats = np.array([])
    desired_lons = np.array([])

    if str(balancing_authority) != "0":
        for plant in range(plant_balancing_authorities.size):
            if (str(plant_balancing_authorities[plant,0])).find(balancing_authority) != -1:
                desired_plant_codes = np.append(desired_plant_codes, plant_codes[plant])
                desired_lats = np.append(desired_lats, lats[plant])
                desired_lons = np.append(desired_lons, lons[plant])

    if str(nerc_region) != "0":
        for plant in range(plant_balancing_authorities.size):
            if (str(plant_nerc_region[plant,0])).find(nerc_region) != -1:
                desired_plant_codes = np.append(desired_plant_codes, plant_codes[plant])
                desired_lats = np.append(desired_lats, lats[plant])
                desired_lons = np.append(desired_lons, lons[plant])

    #find solar plants
    solar_plant_codes = np.array(pd.read_excel(eia_folder+"3_3_Solar_Y2018.xlsx",header=1,usecols=[2]).values)
    solar_capacities = np.array(pd.read_excel(eia_folder+"3_3_Solar_Y2018.xlsx",header=1,usecols=[12]).values)
    solar_status = np.array(pd.read_excel(eia_folder+"3_3_Solar_Y2018.xlsx",header=1,usecols=[7]).values)
    for generator in range(solar_plant_codes.size):
        for code in range(desired_plant_codes.size):
            if desired_plant_codes[code] == solar_plant_codes[generator]:
                if (str(solar_status[generator])).find("OP") != -1:
                    solar_nameplate_capacity = np.append(solar_nameplate_capacity, solar_capacities[generator])
                    solar_lat = np.append(solar_lat, desired_lats[code])
                    solar_lon = np.append(solar_lon, desired_lons[code])

    #find windplants
    wind_plant_codes = np.array(pd.read_excel(eia_folder+"3_2_Wind_Y2018.xlsx",header=1,usecols=[2]).values)
    wind_capacities = np.array(pd.read_excel(eia_folder+"3_2_Wind_Y2018.xlsx",header=1,usecols=[12]).values)
    wind_status = np.array(pd.read_excel(eia_folder+"3_2_Wind_Y2018.xlsx",header=1,usecols=[7]).values)
    for generator in range(wind_plant_codes.size):
        for code in range(desired_plant_codes.size):
            if desired_plant_codes[code] == wind_plant_codes[generator]:
                if (str(wind_status[generator])).find("OP") != -1:
                    wind_nameplate_capacity = np.append(wind_nameplate_capacity, wind_capacities[generator])
                    wind_lat = np.append(wind_lat, desired_lats[code])
                    wind_lon = np.append(wind_lon, desired_lons[code])
    #print("Found "+ str(solar_nameplate_capacity.size + wind_nameplate_capacity.size) +" vg plants")

    solar_efor_array = np.ones(solar_nameplate_capacity.size) * efor_in
    wind_efor_array = np.ones(wind_nameplate_capacity.size) * efor_in

    solar = np.array([solar_nameplate_capacity[:], solar_lat[:], solar_lon[:], solar_efor_array[:]])
    wind  = np.array([wind_nameplate_capacity[:], wind_lat[:], wind_lon[:], wind_efor_array[:]])

    return solar, wind


# Convert the latitude and longitude of the vg into indices for Power Generation to find capacity factor in daily risk
def process_vg(vg, powGen_lats, powGen_lons):
    lats = vg[1,:]
    lons = vg[2,:]
    max_lat = powGen_lats.size - 1
    max_lon = powGen_lons.size - 1

    for i in range(lats.size):
        lat = 0
        if type(lats[i]) is str:
            lats[i] = 0
        while lats[i] > powGen_lats[lat] and lat < max_lat:
            lat += 1
        vg[1,i] = lat


    for i in range(lons.size):
        lon = 0
        if type(lons[i]) is str:
            lons[i] = 0
        while lons[i] > powGen_lons[lon] and lon < max_lon:
            lon += 1
        vg[2,i] = lon

    return vg


# Find hourly contribution from system for a desired number of iterations
def process_system(conventional_system, solar, wind, solar_cf, wind_cf, num_iterations):

    system = np.zeros((8760,num_iterations))

    #conventional contribution
    max_length = 1 #need this for larger systems like wecc, otherwise memory error
    blocks = int(num_iterations/max_length)
    remainder = int(num_iterations)%max_length
    for i in range(blocks):
        conventional = np.array([[conventional_system[0],]*max_length,]*8760)
        conventional_efor = np.array([[conventional_system[1],]*max_length,]*8760)
        conventional = np.sum(conventional * (np.random.random_sample(conventional_efor.shape) > conventional_efor), axis = 2)
        system[:,i*max_length:(i+1)*max_length] = conventional
    if remainder != 0:
        conventional = np.array([[conventional_system[0],]*(remainder),]*8760)
        conventional_efor = np.array([[conventional_system[1],]*(remainder),]*8760)
        conventional = np.sum(conventional * (np.random.random_sample(conventional_efor.shape) > conventional_efor), axis = 2)
        system[:,(blocks*max_length):] = conventional


    del(conventional)
    del(conventional_efor)

    for hour in range(8760):
        capacity = np.zeros(num_iterations)

        #contribution from solar
        lats = np.array(solar[1],dtype=int)
        lons = np.array(solar[2],dtype=int)
        solar_cap = solar[0]*solar_cf[lats,lons,hour]
        solar_cap = np.array([solar_cap,]*num_iterations)
        solar_efor = np.array([solar[3],]*num_iterations)

        capacity = np.sum(solar_cap * (solar_efor < np.random.random_sample(solar_efor.shape)),axis=1)
        del(solar_cap)
        del(solar_efor)

        #contribution from wind
        lats = np.array(wind[1],dtype=int)
        lons = np.array(wind[2],dtype=int)
        wind_cap = wind[0]*wind_cf[lats,lons,hour]
        wind_cap = np.array([wind_cap,]*num_iterations,dtype=float)
        wind_efor = np.array([wind[3],]*num_iterations,dtype=float)

        capacity += np.sum(wind_cap * (wind_efor < np.random.random_sample(wind_efor.shape)),axis=1)
        del(wind_cap)
        del(wind_efor)
        del(lats)
        del(lons)

        system[hour] += capacity
    return system


# Remove the oldest generators from the conventional system
# add an optional oldest year to start at, otherwise set to some low value (e.g. -1)
def remove_oldest(conventional_system, optional_oldest_operating_year):
    oldest_operating_year = np.amin(conventional_system[2,:])
    if optional_oldest_operating_year > oldest_operating_year:
        oldest_operating_year = optional_oldest_operating_year
    erase = np.array([], dtype=int)
    for generator in range(conventional_system[2,:].size):
        if conventional_system[2,generator] <= oldest_operating_year:
            erase = np.append(erase,generator)
    capacity_removed = np.sum(conventional_system[0,erase])
    conventional_system = np.delete(conventional_system,erase,1)
    return conventional_system, oldest_operating_year, capacity_removed


def remove_generators(conventional_system, oldest_year_manual, solar, wind, solar_cf, wind_cf, hourly_peak_load,num_iterations):
    # Get close enough to reliability level
    target_lolp = 0
    total_capacity_removed = 0
    while conventional_system.size > 1 and target_lolp <= 2.4:
        conventional_system, min_operating_year, capacity_removed = remove_oldest(conventional_system, oldest_year_manual)
        system = process_system(conventional_system,solar, wind, solar_cf, wind_cf, 1)
        target_lolp = get_lolp(system, solar_cf, wind_cf, hourly_peak_load, 0, 0, 1)
        total_capacity_removed = total_capacity_removed + capacity_removed

    # Use binary search to add conventional generator until target reliability is reached
    supplement_capacity = capacity_removed / 2.0
    supplement_max = capacity_removed
    supplement_min = 0.0
    conventional_system[0,0] += supplement_capacity
    system = process_system(conventional_system,solar, wind, solar_cf, wind_cf, 100)
    target_lolp = get_lolp(system, solar_cf, wind_cf, hourly_peak_load, 0, 0, 100)

    binary_trial = 0
    while binary_trial < 10 and abs(target_lolp - 2.4) > .01:
        if target_lolp > 2.4:
            conventional_system[0,0] -= supplement_capacity #remove supplement and adjust

            supplement_min = supplement_capacity
            supplement_capacity += (supplement_max - supplement_capacity) / 2
            conventional_system[0,0] += supplement_capacity
        elif target_lolp < 2.4:
            conventional_system[0,0] -= supplement_capacity

            supplement_max = supplement_capacity
            supplement_capacity -= (supplement_capacity - supplement_min) / 2
            conventional_system[0,0] += supplement_capacity
        system = process_system(conventional_system,solar, wind, solar_cf, wind_cf, 100)
        target_lolp = get_lolp(system, solar_cf, wind_cf, hourly_peak_load, 0, 0, 100)
        binary_trial += 1

    total_capacity_removed = total_capacity_removed - supplement_capacity
    #print("Oldest operating year:",min_operating_year)
    #print("Number of active generators:",conventional_system[0].size)
    #print("Capacity removed:",total_capacity_removed)


    return conventional_system


def get_lolp(system, solar_cf, wind_cf, hourly_peak_load, solar_generator, wind_generator, num_iterations):
    lolp = 0.0

    if np.isscalar(solar_generator):
        solar_generator = np.array([[0],[0],[0],[0]])

    if np.isscalar(wind_generator):
        wind_generator = np.array([[0],[0],[0],[0]])

    for hour in range(8760):
        lolp += hourly_risk(hour, system[hour], solar_cf, wind_cf, solar_generator, wind_generator, hourly_peak_load[hour], num_iterations)

    return lolp


def hourly_risk(hour, system, solar_cf, wind_cf, solar_generator, wind_generator, peak_load, num_iterations):

    #contribution from system
    capacity = system

    #contribution from solar_generator
    lats = np.array(solar_generator[1],dtype=int)
    lons = np.array(solar_generator[2],dtype=int)
    solar_generator_cap = solar_generator[0]*solar_cf[lats,lons,hour]
    solar_generator_cap = np.array([solar_generator_cap,]*num_iterations)
    solar_generator_efor = np.array([solar_generator[3],]*num_iterations)

    capacity = capacity + np.sum(solar_generator_cap * (solar_generator_efor < np.random.random_sample(solar_generator_efor.shape)),axis=1)

    #contribution from wind_generator
    lats = np.array(wind_generator[1],dtype=int)
    lons = np.array(wind_generator[2],dtype=int)
    wind_generator_cap = wind_generator[0]*wind_cf[lats,lons,hour]
    wind_generator_cap = np.array([wind_generator_cap,]*num_iterations)
    wind_generator_efor = np.array([wind_generator[3],]*num_iterations)

    capacity = capacity + np.sum(wind_generator_cap * (wind_generator_efor < np.random.random_sample(wind_generator_efor.shape)),axis=1)

    hourly_risk = np.sum(capacity < peak_load) / float(num_iterations)


    return hourly_risk

# use binary search to find elcc (fast)
def get_elcc_binary(system, solar_cf, wind_cf, solar_generator, wind_generator, hourly_peak_load, num_iterations, target_lolp):

    # Use binary search to find elcc of generator(s)
    additional_load =  (np.sum(solar_generator[0]) + np.sum(wind_generator[0])) / 2.0 #MW
    additional_max = np.sum(solar_generator[0]) + np.sum(wind_generator[0])
    additional_min = 0.0

    lolp = get_lolp(system,solar_cf,wind_cf,hourly_peak_load+additional_load,solar_generator,wind_generator,num_iterations)

    binary_trial = 0
    while binary_trial < 20 and abs(target_lolp - lolp) > .005:
        if lolp < target_lolp:
            additional_min = additional_load
            additional_load = additional_load + (additional_max - additional_load) / 2
        if lolp > target_lolp:
            additional_max = additional_load
            additional_load = additional_load - (additional_load - additional_min) / 2
        lolp = get_lolp(system,solar_cf,wind_cf,hourly_peak_load+additional_load,solar_generator,wind_generator,num_iterations)

        binary_trial += 1

    elcc = additional_load
    return elcc

# use sweep search to find elcc (mainly to validate binary)
def get_elcc_sweep(system, solar_cf, wind_cf, solar_generator, wind_generator, hourly_peak_load, num_iterations, target_lolp):

    # Use sweep to find elcc of generator(s)
    additional_load =  0
    additional_max = np.sum(solar_generator[0]) + np.sum(wind_generator[0])

    lolp = get_lolp(system,solar_cf,wind_cf,hourly_peak_load+additional_load,solar_generator,wind_generator,num_iterations)

    additional_load_array = np.array([additional_load])
    lolp_array = np.array([lolp])

    while additional_load < additional_max and lolp < target_lolp:

        additional_load = additional_load + additional_max / 100.
        lolp = get_lolp(system,solar_cf,wind_cf,hourly_peak_load+additional_load,solar_generator,wind_generator,num_iterations)

        additional_load_array = np.append(additional_load_array, additional_load)
        lolp_array = np.append(lolp_array,lolp)


    plt.clf()
    plt.plot(np.array([0,additional_max]), np.array([target_lolp,target_lolp]),c='#777777')
    plt.scatter(additional_load_array,lolp_array,c='#000000')
    plt.scatter(additional_load, lolp,c='#ff0000')
    plt.title("Sweep ELCC Calculation")
    plt.xlabel("Additional Load (MW)")
    plt.ylabel("LOLP (hours per year)")
    filename = "./images/sweep_elcc_.png"
    plt.savefig(filename)
    plt.clf()

    elcc = additional_load
    return elcc


#############################
#            Main           #
#############################


print('{:%Y-%m-%d %H:%M:%S}\tBegin Main'.format(datetime.datetime.now()))

# get file data

powGen_lats, powGen_lons, solar_cf, wind_cf = get_powGen(solar_file, wind_file)

if str(balancing_authority) != "0": #demand file formatted differently for balancing authority/nerc region
    hourly_peak_load = get_balancing_authority_demand_data(demand_file, year)
if str(nerc_region) != "0":
    hourly_peak_load = get_regional_demand_data(demand_file, year)
    oldest_year_manual = 1979

conventional_system = get_conventional_fleet(eia_folder, balancing_authority, nerc_region, year, conventional_efor)
solar, wind = get_vg_system(eia_folder, balancing_authority, nerc_region, year, vg_efor)

# process conventional and vre (derate and correct coordinates for array indexing)

conventional_system = derate(derate_conventional, conventional_system)
solar = process_vg(solar, powGen_lats, powGen_lons)
wind = process_vg(wind, powGen_lats, powGen_lons)

# remove generators to find a target reliability level (2.4 loss of load hours per year)

conventional_system = remove_generators(conventional_system, oldest_year_manual, solar, wind, solar_cf, wind_cf, hourly_peak_load, 100)

print('{:%Y-%m-%d %H:%M:%S}\tSystem Processing'.format(datetime.datetime.now()))
system = process_system(conventional_system,solar,wind,solar_cf,wind_cf,num_iterations)
print('{:%Y-%m-%d %H:%M:%S}\tFinished System Processing'.format(datetime.datetime.now()))

# process generators to be added

solar_generator = np.array([[0],[0],[0],[0]])
wind_generator = np.array([[0],[0],[0],[0]])
if generator_type == "solar":
    solar_generator = np.array([[generator_capacity],[generator_latitude],[generator_longitude],[generator_efor]])
    solar_generator = process_vg(solar_generator, powGen_lats, powGen_lons)
if generator_type == "wind":
    wind_generator = np.array([[generator_capacity],[generator_latitude],[generator_longitude],[generator_efor]])
    wind_generator = process_vg(wind_generator, powGen_lats, powGen_lons)

#complete elcc calculation

target_lolp = get_lolp(system,solar_cf,wind_cf,hourly_peak_load,0,0,num_iterations)
#print("\tTarget LOLP:",target_lolp)
elcc = get_elcc_binary(system,solar_cf,wind_cf,solar_generator,wind_generator,hourly_peak_load,num_iterations,target_lolp)
print("ELCC:", elcc)

print('{:%Y-%m-%d %H:%M:%S}\tFinished Main'.format(datetime.datetime.now()))
