import csv
from datetime import datetime
import math
import os
import os.path
import sys
from os import path
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from numpy import random
import matplotlib
import matplotlib.pyplot as plt

from storage_impl import get_storage_fleet, get_hourly_storage_contribution, make_storage, append_storage

np.random.seed()

# Globals
DEBUG = False
OUTPUT_DIRECTORY = ""

# Get all necessary information from powGen netCDF files: RE capacity factos and corresponding lat/lons
#           Capacity factors in matrix of shape(lats, lon, 8760 hrs) 
def get_powGen(solar_cf_file, wind_cf_file):

    # Error Handling
    if not (path.exists(solar_cf_file) and path.exists(wind_cf_file)):
        error_message = 'Renewable Generation files not available:\n\t'+solar_cf_file+'\n\t'+wind_cf_file
        raise RuntimeError(error_message)
    
    solarPowGen = Dataset(solar_cf_file)
    windPowGen = Dataset(wind_cf_file) #assume solar and wind cover same geographic region

    powGen_lats = np.array(solarPowGen.variables['lat'][:])
    powGen_lons = np.array(solarPowGen.variables['lon'][:])

    cf = dict()
    cf["solar"] = np.array(solarPowGen.variables['ac'][:]) 
    cf["wind"] = np.array(windPowGen.variables['ac'][:])

    solarPowGen.close()
    windPowGen.close()

    return powGen_lats, powGen_lons, cf

# Get hourly load vector
def get_hourly_load(demand_file_in, year, hrsShift=0):
    
    # Open file
    demand_data = pd.read_csv(demand_file_in,delimiter=',',usecols=["date_time","cleaned demand (MW)"],index_col="date_time")

    # Find Given Year
    hourly_load = np.array(demand_data["cleaned demand (MW)"][demand_data.index.str.find(str(year),0,10) != -1].values)

    # Remove leap day
    leap_days=demand_data.index[demand_data.index.str.find("-02-29",0,10) != -1]
    demand_data.drop(leap_days, inplace=True) 
        # two date_time formats from eia cleaned data
    leap_days=demand_data.index[demand_data.index.str.find(str(year)+"0229",0,10) != -1]
    demand_data.drop(leap_days, inplace=True)

    # Shift load
    if hrsShift!=0:
        newLoad = np.array([0]*abs(hrsShift))
        if hrsShift>0:
            hourly_load = np.concatenate([newLoad,hourly_load[:(hourly_load.shape[0]-hrsShift)]])
        else:
            hourly_load = np.concatenate([hourly_load[abs(hrsShift):],newLoad])

    # Error Handling
    if(hourly_load.size != 8760):
        error_message = 'Expected hourly load array of size 8760. Found array of size '+str(hourly_load.size)
        raise RuntimeError(error_message)
    return hourly_load

#loads in hourly temperature for all the coordinates in desired region
def get_temperature_data(temperature_file):
    temperature_data = np.array(Dataset(temperature_file)["T2M"][:][:][:]).T
    return (temperature_data-273.15)

#loads in benchmark fors for temperature incrments of 5 celsius from -15 to 35 for 6 different types of technology
def get_benchmark_fors(benchmark_FORs_file):
    tech_categories = ["Temperature","HD","CC","CT","DS","HD","NU","ST","Other"]
    forData = pd.read_excel(benchmark_FORs_file)    
    benchmark_fors_tech = dict()
    for tech in tech_categories:
        benchmark_fors_tech[tech] = forData[tech].values
    return benchmark_fors_tech

#computes the forced outage rate given an input of temperature and a specific technology type
def calculate_fors(total_efor_array, simplified_tech_list, benchmark_fors,hourly_temp_data):  
    #rounding values to nearest 5 degree due to known for table being given in increments of 5 and rounding to known values
    hourly_temp_data = (5 * np.round(hourly_temp_data/5))
    hourly_temp_data = (np.where(hourly_temp_data > 35,35,hourly_temp_data))
    hourly_temp_data = (np.where(hourly_temp_data < -15,-15,hourly_temp_data))
    
    #finds index of where each rounded temperature would be inserted on temperature array(-15 -> 35)
    temperature_indices = np.searchsorted(benchmark_fors["Temperature"], hourly_temp_data)
    
    for tech in np.unique(simplified_tech_list):
        if tech == '0.0':
            benchmark_for_keyword = "Other"
        else:
            benchmark_for_keyword = tech
        total_efor_array =  np.where(simplified_tech_list == tech,benchmark_fors[benchmark_for_keyword][temperature_indices[:]]/100,total_efor_array)
    
    print("Average annual temperature dependent FOR: " + str(np.average(total_efor_array)))
    return total_efor_array

#create total forced outage rate for all the generators in desired region's fleet
def get_tech_efor_round_downs(simplified_tech_list, latitudes, longitudes,temperature_data,benchmark_fors):
    total_efor_array = np.zeros((len(simplified_tech_list),8760))
    hourly_temp_data = temperature_data[longitudes,latitudes]

    simplified_tech_list = np.array([simplified_tech_list,]*8760).T
    
    return calculate_fors(total_efor_array, simplified_tech_list, benchmark_fors, hourly_temp_data)

#function used to convert technologies of all generators into 6 known for technolgoy realtionships
#if realtionship isnt known treated as constat 5%
def find_desired_tech_indices(desired_tech_list,generator_technology):
    simplified_tech_list = np.zeros(len(generator_technology))
    generator_technology = pd.DataFrame(data=generator_technology.flatten())
    for tech_type in desired_tech_list:
        specific_tech = generator_technology.isin(desired_tech_list[tech_type])
        simplified_tech_list = np.where((generator_technology[specific_tech].fillna(0).values).flatten() != 0,tech_type,simplified_tech_list) 
    return simplified_tech_list

#create main tech list where all the other different types of tech are divided into 6 main known temperature-for relatonships,
# any tech that doesnt fall into 6 groups is given a constant for of .05    
def get_temperature_dependent_efor(latitudes,longitudes,technology,temperature_data,benchmark_fors):
    total_tech_list = dict()
    total_tech_list["CC"] = np.array(["Natural Gas Fired Combined Cycle"])
    total_tech_list["CT"] = np.array(["Natural Gas Steam Turbine","Natural Gas Fired Combustion Turbine","Landfill Gas",])
    total_tech_list["DS"] = np.array(["Natural Gas Internal Combustion Engine"])
    total_tech_list["ST"]  = np.array(["Conventional Steam Coal","Natural Gas Steam Turbine"])
    total_tech_list["NU"]  = np.array(["Nuclear"])
    total_tech_list["HD"]  =  np.array(["Conventional Hydroelectric","Solar Thermal without Energy Storage",
                   "Hydroelectric Pumped Storage","Solar Thermal with Energy Storage","Wood/Wood Waste Biomass"])
    simplified_tech_list = find_desired_tech_indices(total_tech_list,technology)
    #FOR TESTING
    #return simplified_tech_list
    return get_tech_efor_round_downs(simplified_tech_list,latitudes,longitudes,temperature_data,benchmark_fors)

# Implementation of get_conventional_fleet
def get_conventional_fleet_impl(plants, NRE_generators,system_preferences,temperature_data, year,powGen_lats,powGen_lons,benchmark_fors):
    # Remove generators added after simulation year
    active_generators = NRE_generators[NRE_generators["Operating Year"] <= year]
    # Remove renewable generators
    renewable_technologies = np.array(["Solar Photovoltaic", "Onshore Wind Turbine", "Offshore Wind Turbine", "Batteries"])
    active_generators = active_generators[~(active_generators["Technology"].isin(renewable_technologies))] # tilde -> is NOT in

    # Fill empty summer/winter capacities
    active_generators["Summer Capacity (MW)"].where(active_generators["Summer Capacity (MW)"].astype(str) != " ",
                                                    active_generators["Nameplate Capacity (MW)"], inplace=True)
    active_generators["Winter Capacity (MW)"].where(active_generators["Winter Capacity (MW)"].astype(str) != " ", 
                                                    active_generators["Nameplate Capacity (MW)"], inplace=True)
    #getting lats and longs correct indices
    latitudes = find_nearest_impl(plants["Latitude"][active_generators["Plant Code"]].values,powGen_lats)
    longitudes = find_nearest_impl(plants["Longitude"][active_generators["Plant Code"]].values,powGen_lons)
     
    # Convert Dataframe to Dictionary of numpy arrays
    conventional_generators = dict()
    conventional_generators["nameplate"] = active_generators["Nameplate Capacity (MW)"].values
    conventional_generators["summer nameplate"] = active_generators["Summer Capacity (MW)"].values
    conventional_generators["winter nameplate"] = active_generators["Winter Capacity (MW)"].values
    conventional_generators["year"] = active_generators["Operating Year"].values
    conventional_generators["technology"] = active_generators["Technology"].values
    if(system_preferences["temperature dependent FOR"]):
        conventional_generators["efor"] = get_temperature_dependent_efor(latitudes,longitudes, active_generators["Technology"].values,temperature_data,benchmark_fors)
        if not (system_preferences["temperature dependent FOR indpendent of size"]):
            print("Removed temperature dependency FORs for generators smaller then 20 MW")
            conventional_generators["efor"] = np.where(np.array([conventional_generators["nameplate"],]*8760).T <= 20,system_preferences["conventional efor"],conventional_generators["efor"])
    else:
        conventional_generators["efor"] = np.ones(conventional_generators["nameplate"].size) * system_preferences["conventional efor"]                                  
        
    # Error Handling

    if conventional_generators["nameplate"].size == 0:
        error_message = "No existing conventional found."
        raise RuntimeError(error_message)
    
    if DEBUG:
        print("found",conventional_generators["nameplate"].size,"conventional generators")

    return conventional_generators

# Get conventional generators in fleet
def get_conventional_fleet(eia_folder, region, year, system_preferences,powGen_lats,powGen_lons,temperature_data,benchmark_fors):
    # system_preferences

    # Open files
    plants = pd.read_excel(eia_folder+"2___Plant_Y2018.xlsx",skiprows=1,usecols=["Plant Code","NERC Region","Latitude",
                                                                                "Longitude","Balancing Authority Code"])
    all_conventional_generators = pd.read_excel(eia_folder+"3_1_Generator_Y2018.xlsx",skiprows=1,\
                                                    usecols=["Plant Code","Technology","Nameplate Capacity (MW)","Status",
                                                            "Operating Year", "Summer Capacity (MW)", "Winter Capacity (MW)"])
    # Sort by NERC Region and Balancing Authority to filter correct plant codes
    nerc_region_plant_codes = plants["Plant Code"][plants["NERC Region"] == region].values
    balancing_authority_plant_codes = plants["Plant Code"][plants["Balancing Authority Code"] == region].values
    
    desired_plant_codes = np.concatenate((nerc_region_plant_codes, balancing_authority_plant_codes))

    # Error Handling
    if desired_plant_codes.size == 0:
        error_message = "Invalid region/balancing authority: " + region
        raise RuntimeError(error_message)

    # Get operating generators
    active_generators = all_conventional_generators[(all_conventional_generators["Plant Code"].isin(desired_plant_codes)) & (all_conventional_generators["Status"] == "OP")]

    plants.set_index("Plant Code",inplace=True)   
    return get_conventional_fleet_impl(plants,active_generators,system_preferences,temperature_data,year,powGen_lats,powGen_lons,benchmark_fors)

def derate(derate_conventional, conventional_generators):

    if derate_conventional:
        conventional_generators["nameplate"] *= .95
        conventional_generators["summer nameplate"] *= .95
        conventional_generators["winter nameplate"] *= .95
    
# Implementation of get_solar_and_wind_fleet
def get_RE_fleet_impl(plants, RE_generators, desired_plant_codes, year, RE_efor):
    
    # Get operating generators
    active_generators = RE_generators[(RE_generators["Plant Code"].isin(desired_plant_codes)) & (RE_generators["Status"] == "OP")]
    
    # Remove generators added after simulation year
    active_generators = active_generators[active_generators["Operating Year"] <= year]

    # Fill empty summer/winter capacities
    active_generators["Summer Capacity (MW)"].where(active_generators["Summer Capacity (MW)"].astype(str) != " ",
                                                    active_generators["Nameplate Capacity (MW)"], inplace=True)
    active_generators["Winter Capacity (MW)"].where(active_generators["Winter Capacity (MW)"].astype(str) != " ", 
                                                    active_generators["Nameplate Capacity (MW)"], inplace=True)

    # Get coordinates
    latitudes = plants["Latitude"][active_generators["Plant Code"]].values
    longitudes = plants["Longitude"][active_generators["Plant Code"]].values
    # Convert Dataframe to Dictionary of numpy arrays
    RE_generators = dict()
    RE_generators["nameplate"] = active_generators["Nameplate Capacity (MW)"].values
    RE_generators["summer nameplate"] = active_generators["Summer Capacity (MW)"]
    RE_generators["winter nameplate"] = active_generators["Winter Capacity (MW)"]
    RE_generators["lat"] = latitudes
    RE_generators["lon"] = longitudes
    RE_generators["efor"] = np.ones(RE_generators["nameplate"].size) * RE_efor 

    # Error Handling
    if RE_generators["nameplate"].size == 0:
        error_message = "No existing renewables found."
        raise RuntimeWarning(error_message)

    return RE_generators

# Get solar and wind generators in fleet
def get_solar_and_wind_fleet(eia_folder, region, year, RE_efor, powGen_lats, powGen_lons):

    # Open files
    plants = pd.read_excel(eia_folder+"2___Plant_Y2018.xlsx",skiprows=1,usecols=["Plant Code","NERC Region","Latitude",
                                                                                "Longitude","Balancing Authority Code"])
    all_solar_generators = pd.read_excel(eia_folder+"3_3_Solar_Y2018.xlsx",skiprows=1,\
                                usecols=["Plant Code","Nameplate Capacity (MW)",
                                        "Summer Capacity (MW)", "Winter Capacity (MW)",
                                        "Status","Operating Year"])
    all_wind_generators = pd.read_excel(eia_folder+"3_2_Wind_Y2018.xlsx",skiprows=1,\
                                usecols=["Plant Code","Nameplate Capacity (MW)",
                                        "Summer Capacity (MW)", "Winter Capacity (MW)",
                                        "Status","Operating Year"])

     # Sort by NERC Region and Balancing Authority to filter correct plant codes
    nerc_region_plant_codes = plants["Plant Code"][plants["NERC Region"] == region].values
    balancing_authority_plant_codes = plants["Plant Code"][plants["Balancing Authority Code"] == region].values
    
    desired_plant_codes = np.concatenate((nerc_region_plant_codes, balancing_authority_plant_codes))

    # Repeat process for solar and wind
    plants.set_index("Plant Code",inplace=True)
    solar_generators = get_RE_fleet_impl(plants,all_solar_generators,desired_plant_codes,year,RE_efor)
    wind_generators = get_RE_fleet_impl(plants,all_wind_generators,desired_plant_codes,year,RE_efor)

    solar_generators["generator type"] = "solar"
    wind_generators["generator type"] = "wind"

    # Process for lat,lon indices
    solar_generators = get_cf_index(solar_generators,powGen_lats,powGen_lons)
    wind_generators = get_cf_index(wind_generators,powGen_lats,powGen_lons)

    if DEBUG:
        print("found",solar_generators["nameplate"].size+wind_generators["nameplate"].size,"renewable generators")
    
    return solar_generators, wind_generators

# Find index of nearest coordinate. Implementation of get_RE_index
def find_nearest_impl(actual_coordinates, discrete_coordinates):
    
    indices = []
    for coord in actual_coordinates:
        indices.append((np.abs(coord-discrete_coordinates)).argmin())
    return np.array(indices)

# Convert the latitude and longitude of the vg into indices for capacity factor matrix
#
# More detail: The simulated capacity factor maps are of limited resolution. This function
#               identifies the nearest simulated location for renewable energy generators
#               and replaces those generators' latitudes and longitudes with indices for 
#               for the nearest simulated location in the capacity factor maps
def get_cf_index(RE_generators, powGen_lats, powGen_lons):

    RE_generators["lat idx"] = find_nearest_impl(RE_generators["lat"], powGen_lats).astype(int)
    RE_generators["lon idx"] = find_nearest_impl(RE_generators["lon"], powGen_lons).astype(int)

    return RE_generators

# Find expected hourly capacity for RE generators before sampling outages. Of shape (8760 hrs, num generators)
# Implementation of get_hourly_capacity
def get_hourly_RE_impl(RE_generators, cf):

    # combine summer and winter capacities
    RE_winter_nameplate = np.tile(RE_generators["winter nameplate"],(8760//4,1))
    RE_summer_nameplate = np.tile(RE_generators["summer nameplate"],(8760//2,1))
    RE_nameplate = np.vstack((RE_winter_nameplate,RE_summer_nameplate,RE_winter_nameplate))

    # multiply by variable hourly capacity factor
    hours = np.tile(np.arange(8760),(RE_generators["nameplate"].size,1)).T # shape(8760 hrs, num generators)
    RE_capacity = np.multiply(RE_nameplate, cf[RE_generators["lat idx"], RE_generators["lon idx"], hours])
    return RE_capacity

def get_RE_profile_for_storage(cf, *generators):
    renewable_profile = np.zeros(8760)
    for generator in generators:
        renewable_profile = np.add(renewable_profile,np.sum(get_hourly_RE_impl(generator,cf[generator["generator type"]]),axis=1))
    return renewable_profile

# Get hourly capacity matrix for a generator by sampling outage rates over all hours/iterations. Of shape (8760 hrs, num iterations)
# Implementation of get_hourly_capacity
def sample_outages_impl(num_iterations, pre_outage_capacity, generators):

    hourly_capacity = np.zeros((8760,num_iterations))
    # otherwise sample outages and add generator contribution
    max_iterations = 2000 // generators["nameplate"].size # the largest # of iterations to compute at one time (solve memory issues)
    if max_iterations == 0: 
        max_iterations = 1
    for i in range(num_iterations // max_iterations):
        for_matrix = np.random.random_sample((max_iterations,8760,generators["nameplate"].size))>(generators["efor"].T) # shape(its,hours,generators)
        #for_matrix = np.random.random_sample((max_iterations,8760,generators["nameplate"].size))>get_for(generators) # shape(its,hours,generators)
        capacity = np.sum(np.multiply(pre_outage_capacity,for_matrix),axis=2).T # shape(its,hours).T -> shape(hours,its)
        hourly_capacity[:,i*max_iterations:(i+1)*max_iterations] = capacity 
    if num_iterations % max_iterations != 0:
        remaining_iterations = num_iterations % max_iterations
        for_matrix = np.random.random_sample((remaining_iterations,8760,generators["nameplate"].size))>(generators["efor"].T)
        capacity = np.sum(np.multiply(pre_outage_capacity,for_matrix),axis=2).T
        hourly_capacity[:,-remaining_iterations:] = capacity
    return hourly_capacity

# Get the hourly capacity matrix for a set of generators for a desired number of iterations
def get_hourly_capacity(num_iterations, generators, cf=None):
    
    # check for conventional
    if cf is None:
        pre_outage_winter_capacity = np.tile(generators["winter nameplate"],(8760//4,1)) # shape(8760 hrs, num generators)
        pre_outage_summer_capacity = np.tile(generators["summer nameplate"],(8760//2,1))
        pre_outage_capacity = np.vstack((pre_outage_winter_capacity, pre_outage_summer_capacity, pre_outage_winter_capacity))

    # otherwise, renewable source:
    else:
        pre_outage_capacity = get_hourly_RE_impl(generators,cf)

    # sample outages
    hourly_capacity = sample_outages_impl(num_iterations, pre_outage_capacity, generators)

    return hourly_capacity

# Get the hourly capacity matrix for the whole fleet (conventional, solar, and wind)
def get_hourly_fleet_capacity(num_iterations, conventional_generators, solar_generators, 
                                wind_generators, cf, storage_units=None, hourly_load=None, renewable_profile=None):

    hourly_fleet_capacity = np.zeros((8760,num_iterations))

    # conventional, solar, and wind
    hourly_fleet_capacity += get_hourly_capacity(num_iterations,conventional_generators)
    hourly_fleet_capacity += get_hourly_capacity(num_iterations,solar_generators,cf["solar"])
    hourly_fleet_capacity += get_hourly_capacity(num_iterations,wind_generators,cf["wind"])

    if storage_units is not None:
        hourly_fleet_capacity += get_hourly_storage_contribution(   num_iterations,hourly_fleet_capacity,
                                                                    hourly_load,storage_units,renewable_profile)
    
    return hourly_fleet_capacity

# Calculate number of expected hours in which load does not meet demand using monte carlo method
def get_lolh(num_iterations, hourly_capacity, hourly_load):

    # identify where load exceeds capacity (loss-of-load). Of shape(8760 hrs, num iterations)
    lol_matrix = np.where(hourly_load > hourly_capacity.T, 1, 0).T
    hourly_risk = np.sum(lol_matrix,axis=1) / float(num_iterations)
    lolh = np.sum(lol_matrix) / float(num_iterations)
    return lolh, hourly_risk

# Remove the oldest generators from the conventional system
# Implementation of remove_generators
def remove_oldest_impl(generators, manual_oldest_year=0):

    # ignore hydroelectric plants
    not_hydro = generators["technology"] != "Conventional Hydroelectric"

    # find oldest plant
    oldest_year = np.amin(generators["year"][not_hydro]) 

    # check for manual removal
    if manual_oldest_year > oldest_year:
        oldest_year = manual_oldest_year

    # erase all generators older than that year
    erase = np.logical_and(generators["year"] <= oldest_year, not_hydro)
    capacity_removed = np.sum(generators["nameplate"][erase])

    generators["nameplate"] = generators["nameplate"][np.logical_not(erase)]
    generators["summer nameplate"] = generators["summer nameplate"][np.logical_not(erase)]
    generators["winter nameplate"] = generators["winter nameplate"][np.logical_not(erase)]
    generators["year"] = generators["year"][np.logical_not(erase)]
    generators["technology"] = generators["technology"][np.logical_not(erase)]
    generators["efor"] = generators["efor"][np.logical_not(erase)]

    return generators, oldest_year, capacity_removed

def remove_generators_binary_constraints(binary_trial,lolh,target_lolh,num_iterations):
    trial_limit_met = binary_trial < 20
    reliability_met = abs(lolh-target_lolh) > (10 / num_iterations)
    return trial_limit_met and reliability_met

# Remove generators to meet reliability requirement (LOLH of 2.4 by default)
def remove_generators(num_iterations, conventional_generators, solar_generators, wind_generators, storage_units,
                        cf, hourly_load, oldest_year_manual, target_lolh, temperature_dependent_efor):

    # Remove capacity until reliability drops beyond target LOLH/year (low iterations to save time)
    low_iterations = 10
    total_capacity_removed = 0

    # Manual removal
    conventional_generators, oldest_year, capacity_removed = remove_oldest_impl(conventional_generators, oldest_year_manual)

    # get RE profile for storage
    renewable_profile = get_RE_profile_for_storage(cf,solar_generators,wind_generators)

    # Find original reliability
    hourly_fleet_capacity = get_hourly_fleet_capacity(low_iterations,conventional_generators,solar_generators,
                                                        wind_generators,cf,storage_units,hourly_load,renewable_profile)
    lolh, hourly_risk = get_lolh(low_iterations,hourly_fleet_capacity,hourly_load) 
    
    # Error Handling: Under Reliable System
    if lolh >= target_lolh:
        print("LOLH: " + str(lolh))
        error_message = "LOLH already greater than target. Under reliable system."
        oldest_year = "No generators removed"
        print(error_message, lolh)

    while conventional_generators["nameplate"].size > 1 and lolh < target_lolh:
        conventional_generators, oldest_year, capacity_removed = remove_oldest_impl(conventional_generators)
        hourly_fleet_capacity = get_hourly_fleet_capacity(low_iterations,conventional_generators,solar_generators,
                                                            wind_generators,cf,storage_units,hourly_load,renewable_profile)
        lolh, hourly_risk = get_lolh(low_iterations,hourly_fleet_capacity,hourly_load) 
        total_capacity_removed = total_capacity_removed + capacity_removed

        if DEBUG:
            print("Oldest Year:\t",int(oldest_year),"\tLOLH:\t",lolh,capacity_removed)

    # find reliability of higher iteration simulation
    hourly_fleet_capacity = get_hourly_fleet_capacity(num_iterations,conventional_generators,solar_generators,
                                                        wind_generators,cf)

    # create a supplemental unit to match target reliability level
    supplement_capacity = np.sum(conventional_generators["nameplate"]) // 2
    supplement_efor = .07 # reasonable efor for conventional thermal generator
    supplement_generator = make_conventional_generator(supplement_capacity,supplement_efor,temperature_dependent_efor)    
    
    # get hourly capacity of supplemental generator and add to fleet capacity
    hourly_supplement_capacity = get_hourly_capacity(num_iterations,supplement_generator)
    hourly_storage_capacity = get_hourly_storage_contribution(  num_iterations,
                                                                hourly_fleet_capacity+hourly_supplement_capacity, 
                                                                hourly_load, 
                                                                storage_units,
                                                                renewable_profile)
    hourly_total_capacity = hourly_fleet_capacity + hourly_supplement_capacity + hourly_storage_capacity

    # adjust supplement capacity using binary search until reliability is met
    supplement_capacity_max = int(np.sum(conventional_generators["nameplate"]))
    supplement_capacity_min = 0

    lolh, hourly_risk = get_lolh(num_iterations,hourly_total_capacity,hourly_load)

    binary_trial = 0

    while remove_generators_binary_constraints(binary_trial,lolh,target_lolh,num_iterations):
        
        old_supplement_capacity = supplement_capacity

        # under reliable, add supplement capacity
        if lolh > target_lolh: 
            supplement_capacity_min = supplement_capacity
            supplement_capacity += (supplement_capacity_max - supplement_capacity) // 2
        
        # over reliable, remove supplement capacity
        else: 
            supplement_capacity_max = supplement_capacity
            supplement_capacity -= (supplement_capacity - supplement_capacity_min) // 2

        # find new contribution from supplemental generators
        hourly_supplement_capacity = hourly_supplement_capacity / old_supplement_capacity * supplement_capacity
        hourly_storage_capacity = get_hourly_storage_contribution(num_iterations,
                                                            hourly_fleet_capacity+hourly_supplement_capacity, 
                                                            hourly_load, 
                                                            storage_units,
                                                            renewable_profile)
        hourly_total_capacity = hourly_fleet_capacity + hourly_supplement_capacity + hourly_storage_capacity

        # find new lolh
        lolh, hourly_risk = get_lolh(num_iterations,hourly_total_capacity,hourly_load)
        binary_trial += 1
        
        if DEBUG:
            print("Supplement Capacity:\t",int(supplement_capacity),"\tLOLH:\t", lolh)
    
    # adjust supplement capacity
    supplement_generator = make_conventional_generator(supplement_capacity,supplement_efor,temperature_dependent_efor)

    # add supplemental generator to fleet
    conventional_generators = append_conventional_generator(conventional_generators,supplement_generator)

    ############### TESTING ###################
    if DEBUG:
        print(lolh)
        print([i//(30*24) for i in range(len(hourly_risk)) if hourly_risk[i]>0])
        print([(i-7)%24 for i in range(len(hourly_risk)) if hourly_risk[i]>0])
        print([i for i in range(len(hourly_risk)) if hourly_risk[i]>0])
        print([i for i in hourly_risk if i>0])

    total_capacity_removed = total_capacity_removed - supplement_capacity

    print("Oldest operating year:",int(oldest_year))
    print("Number of active generators:",conventional_generators["nameplate"].size)
    print("Supplemental capacity:",supplement_capacity)
    print("Capacity removed:",int(total_capacity_removed))
    print("Conventional fleet average capacity:",(np.sum(conventional_generators["summer nameplate"])+np.sum(conventional_generators["winter nameplate"]))//2)
    print("lolh achieved:",lolh)

    return conventional_generators

def make_conventional_generator(capacity,efor,temperature_dependent_efor,operating_year=9999,generator_type="supplemental"):
    new_generator = dict()

    new_generator["nameplate"] = np.array(capacity)
    new_generator["summer nameplate"] = new_generator["nameplate"]
    new_generator["winter nameplate"] = new_generator["nameplate"]
    new_generator["year"] = np.array(operating_year)
    new_generator["technology"] = np.array(generator_type)

    if temperature_dependent_efor:
        new_generator["efor"] = np.array([efor,]*8760).reshape(1,8760) #reasonable efor for conventional generator
    else:
        new_generator["efor"] = np.array([efor])

    return new_generator
        
def append_conventional_generator(fleet_conventional_generators,additional_generator):
    
    for key in fleet_conventional_generators:
        if key == "efor":
            fleet_conventional_generators[key] = np.concatenate((fleet_conventional_generators[key],additional_generator[key]))
        else:
            fleet_conventional_generators[key] = np.append(fleet_conventional_generators[key],additional_generator[key])

    return fleet_conventional_generators

# move generator parameters into dictionary of numpy arrays (for function compatibility)
def make_RE_generator(generator):

    RE_generator = dict()
    RE_generator["nameplate"] = np.array([generator["nameplate"]])
    RE_generator["summer nameplate"] = np.array([generator["nameplate"]])
    RE_generator["winter nameplate"] = np.array([generator["nameplate"]])
    RE_generator["lat"] = np.array([generator["lat"]])
    RE_generator["lon"] = np.array([generator["lon"]])
    RE_generator["efor"] = np.array([generator["efor"]])
    RE_generator["generator type"] = generator["generator type"]

    return RE_generator

def elcc_binary_constraints(binary_trial, lolh, target_lolh, num_iterations, additional_load, added_capacity):
    trial_limit_met = binary_trial < 20
    reliability_met = abs(lolh - target_lolh) > (10/num_iterations)
    lower_bound_met = additional_load > 1
    upper_bound_met = additional_load < added_capacity - 1
    return trial_limit_met and reliability_met and lower_bound_met and upper_bound_met

# use binary search to find elcc by adjusting additional load
def get_elcc(   num_iterations, hourly_fleet_capacity, hourly_added_generator_capacity,fleet_storage, 
                added_storage, hourly_load, added_capacity, fleet_renewable_profile, added_renewable_profile):

    # find original reliability
    hourly_storage_capacity = get_hourly_storage_contribution(  num_iterations,hourly_fleet_capacity,hourly_load,
                                                                fleet_storage, fleet_renewable_profile)
    hourly_total_capacity = np.add(hourly_fleet_capacity,hourly_storage_capacity)

    if fleet_storage["num units"] != 0:
        dbg_storage = np.array([hourly_load-hourly_fleet_capacity[:,0],hourly_storage_capacity[:,0]]).T
        np.savetxt("storage_dbg.csv",dbg_storage,delimiter=',')

    target_lolh, hourly_risk = get_lolh(num_iterations, hourly_total_capacity, hourly_load)
    print("Target lolh:", target_lolh)
    
    # use binary search to find amount of load needed to match base reliability
    additional_load_max = added_capacity
    additional_load_min = 0
    additional_load = added_capacity / 2.0

    # combine fleet storage with generator storage
    all_storage = append_storage(fleet_storage, added_storage)
    combined_renewable_profile = fleet_renewable_profile + added_renewable_profile

    # include storage operation
    hourly_storage_capacity = get_hourly_storage_contribution(  num_iterations,
                                                                hourly_fleet_capacity+hourly_added_generator_capacity,
                                                                hourly_load+additional_load,
                                                                all_storage, combined_renewable_profile)

    # combine contribution from fleet, RE generator, and added storage
    hourly_total_capacity = hourly_fleet_capacity + hourly_storage_capacity + hourly_added_generator_capacity

    lolh, hourly_risk = get_lolh(num_iterations, hourly_total_capacity, hourly_load + additional_load)
    
    if DEBUG:
            print(lolh)
            print([i%24 for i in range(len(hourly_risk)) if hourly_risk[i]>0])
            print([i for i in hourly_risk if i>0])

    binary_trial = 0

    while elcc_binary_constraints(binary_trial, lolh, target_lolh, num_iterations, additional_load, added_capacity):
        
        #under reliable, remove load
        if lolh > target_lolh: 
            additional_load_max = additional_load
            additional_load -= (additional_load - additional_load_min) / 2.0
        
        # over reliable, add load
        else: 
            additional_load_min = additional_load
            additional_load += (additional_load_max - additional_load) / 2.0
        
        # include storage operation
        hourly_storage_capacity = get_hourly_storage_contribution(  num_iterations,
                                                                    hourly_fleet_capacity+hourly_added_generator_capacity,
                                                                    hourly_load+additional_load,
                                                                    all_storage, combined_renewable_profile)

        # combine contribution from fleet, RE generator, and added storage
        hourly_total_capacity = hourly_fleet_capacity + hourly_storage_capacity + hourly_added_generator_capacity

        # find new lolh
        lolh, hourly_risk = get_lolh(num_iterations, hourly_total_capacity, hourly_load + additional_load)
    
        # print additional debugging information
        if DEBUG:
            print(lolh)
            print([i%24 for i in range(len(hourly_risk)) if hourly_risk[i]>0])
            print([i for i in hourly_risk if i>0])

        binary_trial += 1

    if DEBUG == True:
        print(lolh)
        print([(i//(30*24))+1 for i in range(len(hourly_risk)) if hourly_risk[i]>0])
        print([(i-7)%24 for i in range(len(hourly_risk)) if hourly_risk[i]>0])
        print([i for i in range(len(hourly_risk)) if hourly_risk[i]>0])
        print([i for i in hourly_risk if i>0])

    # Error Handling
    if binary_trial == 20:
        error_message = "Threshold not met in 20 binary trials. lolh: "+str(lolh)
        print(error_message)


    elcc = additional_load

    return elcc, hourly_risk

################ PRINT/SAVE/LOAD ######################

# print all parameters
def print_parameters(*parameters):
    print("Parameters:")
    for sub_parameters in parameters:
        for key, value in sub_parameters.items():
            print("\t",key,":",value)

# save hourly fleet capacity to csv
def save_hourly_fleet_capacity(hourly_fleet_capacity,simulation,system):
    
    # pseudo-unique filename 
    hourly_capacity_filename = str(simulation["year"])+'_'+str(system["region"])+'_'+str(simulation["iterations"])+\
                                str(simulation["target reliability"])+'_'+str(simulation["shift load"])+'_'+\
                                str(system["conventional efor"])+'_'+str(system["renewable efor"])+'.csv'

    # convert to dataframe
    hourly_capacity_df = pd.DataFrame(data=hourly_fleet_capacity,
                            index=np.arange(8760),
                            columns=np.arange(simulation["iterations"]))

    # save to csv
    hourly_capacity_df.to_csv(OUTPUT_DIRECTORY+hourly_capacity_filename)

    return

# load hourly fleet capacity
def load_hourly_fleet_capacity(simulation,system):
    
    # pseudo-unique filename to find h.f.c of previous simulation with similar parameters
    hourly_capacity_filename = str(simulation["year"])+'_'+str(system["region"])+'_'+str(simulation["iterations"])+\
                                str(simulation["target reliability"])+'_'+str(simulation["shift load"])+'_'+\
                                str(system["conventional efor"])+'_'+str(system["renewable efor"])+'.csv'

    if path.exists(hourly_capacity_filename):
        hourly_fleet_capacity =pd.read_csv(hourly_capacity_filename,index_col=0).values
    else:
        error_message = "No saved system found."
        raise RuntimeError(error_message)

    return hourly_fleet_capacity

# save generators to csv
def save_active_generators(conventional, solar, wind):
    
    #conventional
    conventional_generator_array = np.array([conventional["nameplate"],conventional["summer nameplate"],
                                            conventional["winter nameplate"],conventional["year"],
                                            conventional["technology"]])

    conventional_generator_df = pd.DataFrame(data=conventional_generator_array.T,
                                index=np.arange(conventional["nameplate"].size),
                                columns=["Nameplate Capacity (MW)", "Summer Capacity (MW)", 
                                        "Winter Capacity (MW)", "Year", "Technology"])

    conventional_generator_df.to_csv(OUTPUT_DIRECTORY+"active_conventional.csv")

    #solar
    solar_generator_array = np.array([solar["nameplate"],solar["summer nameplate"],
                                    solar["winter nameplate"],solar["lat"],
                                    solar["lon"]])

    solar_generator_df = pd.DataFrame(data=solar_generator_array.T,
                                index=np.arange(solar["nameplate"].size),
                                columns=["Nameplate Capacity (MW)","Summer Capacity (MW)",
                                        "Winter Capacity (MW)","Latitude",
                                        "Longitude"])
    solar_generator_df.to_csv(OUTPUT_DIRECTORY+"active_solar.csv")

    #wind
    wind_generator_array = np.array([wind["nameplate"],wind["summer nameplate"],
                                    wind["winter nameplate"],wind["lat"],
                                    wind["lon"]])

    wind_generator_df = pd.DataFrame(data=wind_generator_array.T,
                                index=np.arange(wind["nameplate"].size),
                                columns=["Nameplate Capacity (MW)","Summer Capacity (MW)",
                                        "Winter Capacity (MW)","Latitude",
                                        "Longitude"])
    wind_generator_df.to_csv(OUTPUT_DIRECTORY+"active_wind.csv")
    return

###################### MAIN ############################

def main(simulation,files,system,generator):
    print("Begin Main:\t",str(datetime.now().time()))

    # initialize global variables
    global DEBUG 
    DEBUG = simulation["print debug"]

    # initialize output 
    global OUTPUT_DIRECTORY
    OUTPUT_DIRECTORY = simulation["output directory"]
    
    # Display parameters
    print_parameters(simulation,files,system,generator)

    aa 

    # get file data
    powGen_lats, powGen_lons, cf = get_powGen(files["solar cf file"],files["wind cf file"])
    hourly_load = get_hourly_load(files["demand file"],simulation["year"],simulation["shift load"]) 
    temperature_data = get_temperature_data(files["temperature file"])
    benchmark_fors = get_benchmark_fors(files["benchmark FORs file"])
    fleet_conventional_generators = get_conventional_fleet(files["eia folder"], simulation["region"],
                                                            simulation["year"], system, powGen_lats, powGen_lons,
                                                            temperature_data, benchmark_fors)
    fleet_solar_generators, fleet_wind_generators = get_solar_and_wind_fleet(files["eia folder"],simulation["region"],
                                                                            simulation["year"], system["renewable efor"],
                                                                            powGen_lats, powGen_lons)
    fleet_storage = get_storage_fleet(  system["fleet storage"],files["eia folder"],simulation["region"],simulation["year"],
                                        system["storage efficiency"],system["storage efor"],system["dispatch strategy"])
    
    # Supplemental fleet_storage
    fleet_supplemental_storage = make_storage(  system["supplemental storage"],system["supplemental storage energy capacity"],
                                                system["supplemental storage power capacity"],system["supplemental storage power capacity"],
                                                system["storage efficiency"],system["storage efor"],system["dispatch strategy"])
    fleet_storage = append_storage(fleet_storage, fleet_supplemental_storage)

    # Save demand 
    np.savetxt(OUTPUT_DIRECTORY+'demand.csv',hourly_load,delimiter=',')
    
    # Load saved system
    if system["setting"] == "load":
        hourly_fleet_capacity = load_hourly_fleet_capacity(simulation,system)

    # Find fleet capacity
    else:
        # derate conventional generators' capacities
        derate(system["derate conventional"],fleet_conventional_generators)

        # remove generators to find a target reliability level (2.4 loss of load hours per year)
        fleet_conventional_generators = remove_generators(simulation["rm generators iterations"],fleet_conventional_generators,
                                                        fleet_solar_generators,fleet_wind_generators,fleet_storage,cf,hourly_load,
                                                        system["oldest year"],simulation["target reliability"],system["temperature dependent FOR"])

        # find hourly capacity
        hourly_fleet_capacity = get_hourly_fleet_capacity(  simulation["iterations"],fleet_conventional_generators,
                                                            fleet_solar_generators,fleet_wind_generators, 
                                                            cf)

    # option to save system for detailed analysis or future simulations
    # filename contains simulation parameters
    if system["setting"] == "save":
        save_hourly_fleet_capacity(hourly_fleet_capacity,simulation,system)  

    # format RE generator 
    RE_generator = make_RE_generator(generator)

    # get cf index
    get_cf_index(RE_generator,powGen_lats,powGen_lons)

    # get hourly capacity matrix
    hourly_RE_generator_capacity = get_hourly_capacity(simulation["iterations"],RE_generator,cf[generator["generator type"]])
    
    # add new generator to profile for storage arbitrage
    # renewable profile for storage arbitrage
    fleet_renewable_profile = get_RE_profile_for_storage(cf,fleet_solar_generators,fleet_wind_generators)
    added_renewable_profile = get_RE_profile_for_storage(cf,fleet_solar_generators,fleet_wind_generators,RE_generator)

    # get added storage
    added_storage = make_storage(   generator["generator storage"],generator["generator storage energy capacity"],
                                    generator["generator storage power capacity"],generator["generator storage power capacity"], 
                                    system["storage efficiency"],system["storage efor"],system["dispatch strategy"])

    # calculate elcc
    added_capacity = generator["nameplate"] + generator["generator storage"]*generator["generator storage power capacity"]
    elcc, hourlyRisk = get_elcc(    simulation["iterations"],hourly_fleet_capacity,hourly_RE_generator_capacity, 
                                    fleet_storage,added_storage, hourly_load, added_capacity, 
                                    fleet_renewable_profile, added_renewable_profile)
    print(added_capacity)
    print("**********!!!!!!!!!!!!***********ELCC:", int(elcc/added_capacity*100))

    if DEBUG:
        save_active_generators(fleet_conventional_generators,fleet_solar_generators,fleet_wind_generators)
        np.savetxt(OUTPUT_DIRECTORY+'hourlyRisk.csv',hourlyRisk,delimiter=',')

    print("End Main:\t",str(datetime.now().time()))
    return elcc,hourlyRisk