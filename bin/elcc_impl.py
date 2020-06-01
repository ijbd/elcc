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

np.random.seed()

# Globals
DEBUG = False
TIMESTAMPS = None
OUTPUT_FOLDER = ""

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
def get_demand_data(demand_file_in, year, hrsShift=0):
    
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


# Get conventional generators in fleet
def get_conventional_fleet(eia_folder, region, year, conventional_efor):
    
    # Open files
    plants = pd.read_excel(eia_folder+"2___Plant_Y2018.xlsx",skiprows=1,usecols=["Plant Code","NERC Region","Balancing Authority Code"])
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

    # Remove generators added after simulation year
    active_generators = active_generators[active_generators["Operating Year"] <= year]

    # Remove renewable generators
    renewable_technologies = np.array(["Solar Photovoltaic", "Onshore Wind Turbine", "Offshore Wind Turbine", "Batteries"])
    active_generators = active_generators[~(active_generators["Technology"].isin(renewable_technologies))] # tilde -> is NOT in

    # Fill empty summer/winter capacities
    active_generators["Summer Capacity (MW)"].where(active_generators["Summer Capacity (MW)"].astype(str) != " ",
                                                    active_generators["Nameplate Capacity (MW)"], inplace=True)
    active_generators["Winter Capacity (MW)"].where(active_generators["Winter Capacity (MW)"].astype(str) != " ", 
                                                    active_generators["Nameplate Capacity (MW)"], inplace=True)
    

    # Convert Dataframe to Dictionary of numpy arrays
    conventional_generators = dict()
    conventional_generators["nameplate"] = active_generators["Nameplate Capacity (MW)"].values
    conventional_generators["summer nameplate"] = active_generators["Summer Capacity (MW)"].values
    conventional_generators["winter nameplate"] = active_generators["Winter Capacity (MW)"].values
    conventional_generators["year"] = active_generators["Operating Year"].values
    conventional_generators["type"] = active_generators["Technology"].values
    conventional_generators["efor"] = np.ones(conventional_generators["nameplate"].size) * conventional_efor

    # Error Handling
    if conventional_generators["nameplate"].size == 0:
        error_message = "No existing conventional found."
        raise RuntimeError(error_message)
    
    if DEBUG:
        print("found",conventional_generators["nameplate"].size,"conventional generators")

    return conventional_generators


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
def get_solar_and_wind_fleet(eia_folder, region, year, RE_efor):

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


# Get hourly capacity matrix for a generator by sampling outage rates over all hours/iterations. Of shape (8760 hrs, num iterations)
# Implementation of get_hourly_capacity
def sample_outages_impl(num_iterations, pre_outage_capacity, generators):

    hourly_capacity = np.zeros((8760,num_iterations))

    # otherwise sample outages and add generator contribution
    max_iterations = 2000 // generators["nameplate"].size # the largest # of iterations to compute at one time (solve memory issues)
    if max_iterations == 0: 
        max_iterations = 1 

    for i in range(num_iterations // max_iterations):
        for_matrix = np.random.random_sample((max_iterations,8760,generators["nameplate"].size))>generators["efor"] # shape(its,hours,generators)
        capacity = np.sum(np.multiply(pre_outage_capacity,for_matrix),axis=2).T # shape(its,hours).T -> shape(hours,its)
        hourly_capacity[:,i*max_iterations:(i+1)*max_iterations] = capacity 

        if DEBUG:
            print(i,"of",num_iterations // max_iterations,"blocks complete")

    if num_iterations % max_iterations != 0:
        remaining_iterations = num_iterations % max_iterations
        for_matrix = np.random.random_sample((remaining_iterations,8760,generators["nameplate"].size))>generators["efor"]
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
def get_hourly_fleet_capacity(num_iterations, conventional_generators, solar_generators, wind_generators, cf):

    hourly_fleet_capacity = np.zeros((8760,num_iterations))

    # conventional, solar, and wind
    hourly_fleet_capacity += get_hourly_capacity(num_iterations,conventional_generators)
    hourly_fleet_capacity += get_hourly_capacity(num_iterations,solar_generators,cf["solar"])
    hourly_fleet_capacity += get_hourly_capacity(num_iterations,wind_generators,cf["wind"])

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
    not_hydro = generators["type"] != "Conventional Hydroelectric"

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
    generators["type"] = generators["type"][np.logical_not(erase)]
    generators["efor"] = generators["efor"][np.logical_not(erase)]

    return generators, oldest_year, capacity_removed


# Remove generators to meet reliability requirement (LOLH of 2.4 by default)
def remove_generators(num_iterations, conventional_generators, solar_generators, wind_generators, cf,
                        hourly_load, oldest_year_manual, target_lolh):

    # Remove capacity until reliability drops beyond target LOLH/year (low iterations to save time)
    low_iterations = 10
    total_capacity_removed = 0

    # Find original reliability
    hourly_fleet_capacity = get_hourly_fleet_capacity(low_iterations,conventional_generators,solar_generators,wind_generators,cf)
    lolh, hourly_risk = get_lolh(low_iterations,hourly_fleet_capacity,hourly_load) 
    
    # Error Handling: Under Reliable System
    if lolh >= target_lolh:
        error_message = "LOLH already greater than target. Under reliable system."
        oldest_year = "No generators removed"
        print(error_message, lolh)

    while conventional_generators["nameplate"].size > 1 and lolh < target_lolh:
        conventional_generators, oldest_year, capacity_removed = remove_oldest_impl(conventional_generators, oldest_year_manual)
        hourly_fleet_capacity = get_hourly_fleet_capacity(low_iterations,conventional_generators,solar_generators,wind_generators,cf)
        lolh, hourly_risk = get_lolh(low_iterations,hourly_fleet_capacity,hourly_load) 
        total_capacity_removed = total_capacity_removed + capacity_removed

        if DEBUG:
            print(oldest_year,lolh,capacity_removed)

    # find reliability of higher iteration simulation
    hourly_fleet_capacity = get_hourly_fleet_capacity(num_iterations,conventional_generators,solar_generators,wind_generators,cf)
    lolh, hourly_risk = get_lolh(num_iterations,hourly_fleet_capacity,hourly_load) 

    # create a supplemental unit to match target reliability level
    supplement_generator = dict()

    supplement_capacity = np.sum(conventional_generators["nameplate"]) // 2
    supplement_generator["nameplate"] = np.array([supplement_capacity])
    supplement_generator["summer nameplate"] = supplement_generator["nameplate"]
    supplement_generator["winter nameplate"] = supplement_generator["nameplate"]
    supplement_generator["efor"] = np.array([.07]) #reasonable efor for conventional generator

    # get hourly capacity of supplemental generator and add to fleet capacity
    hourly_supplement_capacity = get_hourly_capacity(num_iterations,supplement_generator)
    hourly_total_capacity = np.add(hourly_fleet_capacity, hourly_supplement_capacity)

    # adjust supplement capacity using binary search until reliability is met
    supplement_capacity_max = int(np.sum(conventional_generators["nameplate"]))
    supplement_capacity_min = 0
    binary_trial = 0

    lolh, hourly_risk = get_lolh(num_iterations,hourly_total_capacity,hourly_load)

    while binary_trial < 20 and abs(lolh-target_lolh) > (10 / num_iterations):
        
        # under reliable, add supplement capacity
        if lolh > target_lolh: 
            supplement_capacity_min = supplement_capacity
            supplement_capacity += (supplement_capacity_max - supplement_capacity) // 2
        
        # over reliable, remove supplement capacity
        else: 
            supplement_capacity_max = supplement_capacity
            supplement_capacity -= (supplement_capacity - supplement_capacity_min) // 2

        # adjust supplement capacity
        supplement_generator["nameplate"] = np.array([supplement_capacity])
        supplement_generator["summer nameplate"] = supplement_generator["nameplate"]
        supplement_generator["winter nameplate"] = supplement_generator["nameplate"]

        # find new contribution from supplemental generators
        hourly_supplement_capacity = get_hourly_capacity(num_iterations,supplement_generator)
        hourly_total_capacity = hourly_fleet_capacity + hourly_supplement_capacity

        # find new lolh
        lolh, hourly_risk = get_lolh(num_iterations,hourly_total_capacity,hourly_load)
        binary_trial += 1

    # add supplemental generator to fleet
    conventional_generators["nameplate"] = np.append(conventional_generators["nameplate"], supplement_generator["nameplate"])
    conventional_generators["summer nameplate"] = np.append(conventional_generators["summer nameplate"], supplement_generator["summer nameplate"])
    conventional_generators["winter nameplate"] = np.append(conventional_generators["winter nameplate"], supplement_generator["winter nameplate"])
    conventional_generators["efor"] = np.append(conventional_generators["efor"], supplement_generator["efor"])
    conventional_generators["year"] = np.append(conventional_generators["year"], 9999) 
    conventional_generators["type"] = np.append(conventional_generators["type"], "supplemental")

    ############### TESTING ###################
    if DEBUG:
        print(lolh)
        print([i//(30*24) for i in range(len(hourly_risk)) if hourly_risk[i]>0])
        print([(i-7)%24 for i in range(len(hourly_risk)) if hourly_risk[i]>0])
        print([i for i in range(len(hourly_risk)) if hourly_risk[i]>0])
        print([i for i in hourly_risk if i>0])

    total_capacity_removed = total_capacity_removed - supplement_capacity

    print("Oldest operating year:",oldest_year)
    print("Number of active generators:",conventional_generators["nameplate"].size)
    print("Supplemental capacity:",supplement_capacity)
    print("Capacity removed:",int(total_capacity_removed))
    print("Conventional fleet average capacity:",(np.sum(conventional_generators["summer nameplate"])+np.sum(conventional_generators["winter nameplate"]))//2)
    print("lolh achieved:",lolh)

    return conventional_generators


# move generator parameters into dictionary of numpy arrays (for function compatibility)
def get_RE_generator(generator):
    RE_generator = dict()
    RE_generator["nameplate"] = np.array([generator["nameplate"]])
    RE_generator["summer nameplate"] = np.array([generator["nameplate"]])
    RE_generator["winter nameplate"] = np.array([generator["nameplate"]])
    RE_generator["lat"] = np.array([generator["lat"]])
    RE_generator["lon"] = np.array([generator["lon"]])
    RE_generator["efor"] = np.array([generator["efor"]])
    return RE_generator


# use binary search to find elcc by adjusting additional load
def get_elcc(num_iterations, hourly_fleet_capacity, hourly_RE_generator_capacity, hourly_load, RE_generator_nameplate):

    print('getting ELCC!')

    # find original reliability
    target_lolh, hourly_risk = get_lolh(num_iterations, hourly_fleet_capacity, hourly_load)

    print("Target lolh:", target_lolh)

    # combine contribution from fleet and RE generator
    hourly_total_capacity = np.add(hourly_fleet_capacity, hourly_RE_generator_capacity)

    # use binary search to find amount of load needed to match base reliability
    additional_load_max = RE_generator_nameplate
    additional_load_min = 0
    additional_load = RE_generator_nameplate / 2.0

    lolh, hourly_risk = get_lolh(num_iterations, hourly_total_capacity, hourly_load + additional_load)
    
    binary_trial = 0
    while binary_trial < 20 and abs(lolh - target_lolh) > (10/num_iterations):
        
        #under reliable, remove load
        if lolh > target_lolh: 
            additional_load_max = additional_load
            additional_load -= (additional_load - additional_load_min) / 2.0
        
        # over reliable, add load
        else: 
            additional_load_min = additional_load
            additional_load += (additional_load_max - additional_load) / 2.0
        
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
        raise RuntimeWarning(error_message)

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
                                str(simulation["target lolh"])+'_'+str(simulation["shift load"])+'_'+\
                                str(system["conventional efor"])+'_'+str(system["RE efor"])+'.csv'

    # convert to dataframe
    hourly_capacity_df = pd.DataFrame(data=hourly_fleet_capacity,
                            index=np.arange(8760),
                            columns=np.arange(simulation["iterations"]))

    # save to csv
    hourly_capacity_df.to_csv(OUTPUT_FOLDER+hourly_capacity_filename)

    return


# load hourly fleet capacity
def load_hourly_fleet_capacity(simulation,system):
    
    # pseudo-unique filename to find h.f.c of previous simulation with similar parameters
    hourly_capacity_filename = str(simulation["year"])+'_'+str(system["region"])+'_'+str(simulation["iterations"])+\
                                str(simulation["target lolh"])+'_'+str(simulation["shift load"])+'_'+\
                                str(system["conventional efor"])+'_'+str(system["RE efor"])+'.csv'

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
                                            conventional["type"],conventional["efor"]])

    conventional_generator_df = pd.DataFrame(data=conventional_generator_array.T,
                                index=np.arange(conventional["nameplate"].size),
                                columns=["Nameplate Capacity (MW)", "Summer Capacity (MW)", 
                                        "Winter Capacity (MW)", "Year", "Type", "EFOR"])

    conventional_generator_df.to_csv(OUTPUT_FOLDER+"active_conventional.csv")

    #solar
    solar_generator_array = np.array([solar["nameplate"],solar["summer nameplate"],
                                    solar["winter nameplate"],solar["lat"],
                                    solar["lon"],solar["efor"]])

    solar_generator_df = pd.DataFrame(data=solar_generator_array.T,
                                index=np.arange(solar["nameplate"].size),
                                columns=["Nameplate Capacity (MW)","Summer Capacity (MW)",
                                        "Winter Capacity (MW)","Latitude",
                                        "Longitude","EFOR"])
    solar_generator_df.to_csv(OUTPUT_FOLDER+"active_solar.csv")

    #wind
    wind_generator_array = np.array([wind["nameplate"],wind["summer nameplate"],
                                    wind["winter nameplate"],wind["lat"],
                                    wind["lon"],wind["efor"]])

    wind_generator_df = pd.DataFrame(data=wind_generator_array.T,
                                index=np.arange(wind["nameplate"].size),
                                columns=["Nameplate Capacity (MW)","Summer Capacity (MW)",
                                        "Winter Capacity (MW)","Latitude",
                                        "Longitude","EFOR"])
    wind_generator_df.to_csv(OUTPUT_FOLDER+"active_wind.csv")
    return


################### RUNTIME STATS #######################


def init_timestamps():
    global TIMESTAMPS
    first_timestamp = [[str(datetime.now().time()), "main(begin)"]]
    TIMESTAMPS =  pd.DataFrame(first_timestamp,columns=["Timestamp", "Event"])
    

def add_timestamp(event):
    global TIMESTAMPS 
    # add new event
    new_timestamp = pd.DataFrame([[str(datetime.now().time()), event]],columns=["Timestamp", "Event"])
    TIMESTAMPS = TIMESTAMPS.append(new_timestamp)


def save_timestamps():
    global TIMESTAMPS
    TIMESTAMPS.set_index("Timestamp")
    TIMESTAMPS.to_csv(OUTPUT_FOLDER+"runtime_profile.csv",index=False)


##################### MAIN ##########################


def main(simulation,files,system,generator):
    
    ####### SET-UP ######

    # initialize global variables
    global DEBUG 
    DEBUG = simulation["debug"]

    # initialize output 
    global OUTPUT_FOLDER
    OUTPUT_FOLDER = simulation["output folder"]
    
    # Display params
    print_parameters(simulation,files,system,generator)

    ###### PROGRAM ######

    # get file data
    powGen_lats, powGen_lons, cf = get_powGen(files["solar cf file"],files["wind cf file"])
    hourly_load = get_demand_data(files["demand file"],simulation["year"],simulation["shift load"]) 

    # save demand 
    np.savetxt(OUTPUT_FOLDER+'demand.csv',hourly_load,delimiter=',')
    
    # Load saved system
    if system["setting"] == "load":
        hourly_fleet_capacity = load_hourly_fleet_capacity(simulation,system)

    # Find fleet 
    else:
        # Get conventional system (nameplate capacity, year, technology, and efor)
        fleet_conventional_generators = get_conventional_fleet(files["eia folder"],simulation["region"],
                                                                simulation["year"],system["conventional efor"])

        # Get RE system (nameplate capacity, latitude, longitude, and efor)
        fleet_solar_generators, fleet_wind_generators = get_solar_and_wind_fleet(files["eia folder"],simulation["region"],
                                                                                simulation["year"], system["RE efor"])

        # add lat/lon indices for cf matrix
        get_cf_index(fleet_solar_generators, powGen_lats, powGen_lons)
        get_cf_index(fleet_wind_generators, powGen_lats, powGen_lons)

        # derate conventional generators' capacities
        derate(system["derate conventional"],fleet_conventional_generators)

        # remove generators to find a target reliability level (2.4 loss of load hours per year)
        fleet_conventional_generators = remove_generators(simulation["rm generators iterations"],fleet_conventional_generators,
                                                            fleet_solar_generators,fleet_wind_generators,cf,hourly_load,system["oldest year"],
                                                            simulation["target lolh"])

        # find hourly capacity
        hourly_fleet_capacity = get_hourly_fleet_capacity(simulation["iterations"],fleet_conventional_generators,
                                                            fleet_solar_generators,fleet_wind_generators,cf)

    # option to save system for detailed analysis or future simulations
    # filename contains simulation parameters
    if system["setting"] == "save":
        save_hourly_fleet_capacity(hourly_fleet_capacity,simulation,system)  
        
        

    # format RE generator and get hourly capacity matrix
    RE_generator = get_RE_generator(generator)

    # get cf index
    get_cf_index(RE_generator,powGen_lats,powGen_lons)

    # get hourly capacity matrix
    hourly_RE_generator_capacity = get_hourly_capacity(simulation["iterations"],RE_generator,cf[generator["type"]])
    
    # calculate elcc
    elcc, hourlyRisk = get_elcc(simulation["iterations"],hourly_fleet_capacity,hourly_RE_generator_capacity,
                                    hourly_load, generator["nameplate"])

    print("******!!!!!!!!!!!!***********ELCC:", int(elcc/generator["nameplate"]*100))
    add_timestamp("main(end)")  
    
    if DEBUG:
        save_active_generators(fleet_conventional_generators,fleet_solar_generators,fleet_wind_generators)
        np.savetxt(OUTPUT_FOLDER+'hourlyRisk.csv',hourlyRisk,delimiter=',')
        save_timestamps()
    return elcc,hourlyRisk