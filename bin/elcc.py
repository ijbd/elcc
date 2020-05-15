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
import matplotlib

np.random.seed()

# Get all necessary information from powGen netCDF files: RE capacity factos and corresponding lat/lons
#           Capacity factors in matrix of shape(lats, lon, 8760 hrs) 
def get_powGen(solar_cf_file, wind_cf_file):
    solarPowGen = Dataset(solar_cf_file)
    windPowGen = Dataset(wind_cf_file) #assume solar and wind cover same geographic region

    powGen_lats = np.array(solarPowGen.variables['lat'][:])
    powGen_lons = np.array(solarPowGen.variables['lon'][:])

    cf = dict()
    cf["solar"] = np.array(solarPowGen.variables['ac'][:]) 
    cf["wind"] = np.array(windPowGen.variables['ac'][:])

    solarPowGen.close()
    windPowGen.close()

    # Error Handling
    if cf["solar"].shape != (powGen_lats.size, powGen_lons.size, 8760):
        error_message = 'Expected capacity factor array of shape ('+str(powGen_lats.size)+','+str(powGen_lons.size)+'8760). Found array of shape '+str(cf["solar"].shape)
        raise RuntimeError(error_message)

    return powGen_lats, powGen_lons, cf


# Get hourly load vector
def get_demand_data(demand_file_in, year_in, hrsShift=0):
    demand_data = pd.read_csv(demand_file_in,delimiter=',',usecols=["date_time","cleaned demand (MW)"],index_col="date_time")

    # Remove Leap Days
    leap_days=demand_data.index[demand_data.index.str.find("-02-29",0,10) != -1]
    demand_data.drop(leap_days, inplace=True) 
        # two date_time formats from eia cleaned data
    leap_days=demand_data.index[demand_data.index.str.find(str(year_in)+"0229",0,10) != -1]
    demand_data.drop(leap_days, inplace=True)

    # Find Given Year
    hourly_load = np.array(demand_data["cleaned demand (MW)"][demand_data.index.str.find(str(year_in),0,10) != -1].values)

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
def get_conventional_fleet(eia_folder, region, year_in, derate_conventional, conventional_efor):
    
    # Open files
    plants = pd.read_excel(eia_folder+"2___Plant_Y2018.xlsx",skiprows=1,usecols=["Plant Code","NERC Region","Balancing Authority Code"])
    all_conventional_generators = pd.read_excel(eia_folder+"3_1_Generator_Y2018.xlsx",skiprows=1,\
                                usecols=["Plant Code","Technology","Nameplate Capacity (MW)","Status","Operating Year"])

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

    # Remove renewable generators
    renewable_technologies = np.array(["Solar Photovoltaic", "Onshore Wind Turbine", "Offshore Wind Turbine", "Batteries"])
    active_generators = active_generators[~(active_generators["Technology"].isin(renewable_technologies))] # tilde -> is NOT in

    # Convert Dataframe to Dictionary of numpy arrays
    conventional_generators = dict()
    conventional_generators["nameplate"] = active_generators["Nameplate Capacity (MW)"].values
    conventional_generators["year"] = active_generators["Operating Year"].values
    conventional_generators["type"] = active_generators["Technology"].values
    conventional_generators["efor"] = np.ones(conventional_generators["nameplate"].size) * conventional_efor

    # Derate (optional)
    if derate_conventional:
        conventional_generators["nameplate"] *= .95

    # Error Handling
    if conventional_generators["nameplate"].size == 0:
        error_message = "No existing conventional found."
        raise RuntimeError(error_message)

    return conventional_generators


# Implementation of get_solar_and_wind_fleet
def get_RE_fleet_impl(plants, RE_generators, desired_plant_codes, RE_efor):
   
    # Get operating generators
    active_generators = RE_generators[(RE_generators["Plant Code"].isin(desired_plant_codes)) & (RE_generators["Status"] == "OP")]
    
    # Get coordinates
    latitudes = plants["Latitude"][active_generators["Plant Code"]].values
    longitudes = plants["Longitude"][active_generators["Plant Code"]].values

    # Convert Dataframe to Dictionary of numpy arrays
    RE_generators = dict()
    RE_generators["nameplate"] = active_generators["Nameplate Capacity (MW)"].values
    RE_generators["lat"] = latitudes
    RE_generators["lon"] = longitudes
    RE_generators["efor"] = np.ones(RE_generators["nameplate"].size) * RE_efor 

    # Error Handling
    if RE_generators["nameplate"].size == 0:
        error_message = "No existing renewables found."
        raise RuntimeWarning(error_message)

    return RE_generators


# Get solar and wind generators in fleet
def get_solar_and_wind_fleet(eia_folder, region, year_in, RE_efor):

    # Open files
    plants = pd.read_excel(eia_folder+"2___Plant_Y2018.xlsx",skiprows=1,usecols=["Plant Code","NERC Region","Latitude","Longitude","Balancing Authority Code"])
    all_solar_generators = pd.read_excel(eia_folder+"3_3_Solar_Y2018.xlsx",skiprows=1,\
                                usecols=["Plant Code","Nameplate Capacity (MW)","Status"])
    all_wind_generators = pd.read_excel(eia_folder+"3_2_Wind_Y2018.xlsx",skiprows=1,\
                                usecols=["Plant Code","Nameplate Capacity (MW)","Status"])

     # Sort by NERC Region and Balancing Authority to filter correct plant codes
    nerc_region_plant_codes = plants["Plant Code"][plants["NERC Region"] == region].values
    balancing_authority_plant_codes = plants["Plant Code"][plants["Balancing Authority Code"] == region].values
    
    desired_plant_codes = np.concatenate((nerc_region_plant_codes, balancing_authority_plant_codes))

    # Repeat process for solar and wind
    plants.set_index("Plant Code",inplace=True)
    solar_generators = get_RE_fleet_impl(plants,all_solar_generators,desired_plant_codes,RE_efor)
    wind_generators = get_RE_fleet_impl(plants,all_wind_generators,desired_plant_codes,RE_efor)

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
    RE_nameplate = RE_generators["nameplate"]
    hrs = np.array([np.arange(8760),]*RE_nameplate.size).T
    RE_capacity = np.multiply(RE_nameplate, cf[RE_generators["lat idx"], RE_generators["lon idx"], hrs])
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
        pre_outage_capacity = np.array([generators["nameplate"],]*8760) # shape(8760 hrs, num generators)
    
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
    generators["year"] = generators["year"][np.logical_not(erase)]
    generators["type"] = generators["type"][np.logical_not(erase)]
    generators["efor"] = generators["efor"][np.logical_not(erase)]

    return generators, oldest_year, capacity_removed


# Remove generators to meet reliability requirement (LOLH of 2.4 by default)
def remove_generators(num_iterations, conventional_generators, solar_generators, wind_generators, cf,
                        hourly_load, oldest_year_manual, target_lolh):

    # Remove capacity until reliability drops beyond target LOLH/year (low iterations to save time)
    lolh = 0
    low_iterations = 10
    total_capacity_removed = 0
    
    while conventional_generators["nameplate"].size > 1 and lolh <= target_lolh*1.1:
        conventional_generators, oldest_year, capacity_removed = remove_oldest_impl(conventional_generators, oldest_year_manual)
        hourly_fleet_capacity = get_hourly_fleet_capacity(low_iterations,conventional_generators,solar_generators,wind_generators,cf)
        lolh, hourly_risk = get_lolh(low_iterations,hourly_fleet_capacity,hourly_load) 
        total_capacity_removed = total_capacity_removed + capacity_removed

    # find reliability of higher iteration simulation
    hourly_fleet_capacity = get_hourly_fleet_capacity(num_iterations,conventional_generators,solar_generators,wind_generators,cf)
    lolh, hourly_risk = get_lolh(num_iterations,hourly_fleet_capacity,hourly_load) 

    # create a supplemental unit to match target reliability level
    supplement_generator = dict()

    supplement_capacity = total_capacity_removed / 2.0
    supplement_generator["nameplate"] = np.array([supplement_capacity])
    supplement_generator["efor"] = np.array([.07]) #reasonable efor for conventional generator

    # get hourly capacity of supplemental generator and add to fleet capacity
    hourly_supplement_capacity = get_hourly_capacity(num_iterations,supplement_generator)
    hourly_total_capacity = np.add(hourly_fleet_capacity, hourly_supplement_capacity)

    # adjust supplement capacity using binary search until reliability is met
    supplement_capacity_max = total_capacity_removed
    supplement_capacity_min = 0.0
    binary_trial = 0

    lolh, hourly_risk = get_lolh(num_iterations,hourly_total_capacity,hourly_load)

    while binary_trial < 20 and abs(lolh-target_lolh) > (10 / num_iterations):
        
        # under reliable, add supplement capacity
        if lolh > target_lolh: 
            supplement_capacity_min = supplement_capacity
            supplement_capacity += (supplement_capacity_max - supplement_capacity) / 2.0
        
        # over reliable, remove supplement capacity
        else: 
            supplement_capacity_max = supplement_capacity
            supplement_capacity -= (supplement_capacity - supplement_capacity_min) / 2.0

        # adjust supplement capacity and 
        supplement_generator["nameplate"] = np.array([supplement_capacity])
        hourly_supplement_capacity = get_hourly_capacity(num_iterations,supplement_generator)
        hourly_total_capacity = hourly_fleet_capacity + hourly_supplement_capacity

        # find new lolh
        lolh, hourly_risk = get_lolh(num_iterations,hourly_total_capacity,hourly_load)
        binary_trial += 1

    # add supplemental generator to fleet
    conventional_generators["nameplate"] = np.append(conventional_generators["nameplate"], supplement_generator["nameplate"])
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
    print("Capacity removed:",total_capacity_removed)
    print("Conventional fleet capacity:",np.sum(conventional_generators["nameplate"]))
    print("lolh achieved:",lolh)

    return conventional_generators


# move generator parameters into dictionary of numpy arrays (for function compatibility)
def get_RE_generator(generator):
    RE_generator = dict()
    RE_generator["nameplate"] = np.array([generator["nameplate"]])
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


################ SAVE/LOAD ######################


# save hourly fleet capacity to csv
def save_hourly_fleet_capacity(hourly_fleet_capacity,simulation,system,generator):
    
    # pseudo-unique filename 
    hourly_capacity_filename = str(simulation["year"])+'_'+str(simulation["iterations"])+'_'+str(system["region"])+\
                    '_'+str(generator["nameplate"])+'_'+generator["type"]+'_'+str(generator["lat"])+'_'+str(generator["lon"])+'.csv'

    # convert to dataframe
    hourly_capacity_df = pd.DataFrame(data=hourly_fleet_capacity,
                            index=np.arange(8760),
                            columns=np.arange(simulation["iterations"]))

    # save to csv
    hourly_capacity_df.to_csv(hourly_capacity_filename)

    return


# load hourly fleet capacity
def load_hourly_fleet_capacity(simulation,system,generator):
    
    # pseudo-unique filename to find h.f.c of previous simulation with similar parameters
    hourly_capacity_filename = str(simulation["year"])+'_'+str(simulation["iterations"])+'_'+str(system["region"])+\
                        '_'+str(generator["nameplate"])+'_'+generator["type"]+'_'+str(generator["lat"])+'_'+str(generator["lon"])+'.csv'
    
    if path.exists(hourly_capacity_filename):
        hourly_fleet_capacity =pd.read_csv(hourly_capacity_filename,index_col=0).values
    else:
        error_message = "No saved system found."
        raise RuntimeError(error_message)

    return hourly_fleet_capacity


# save generators to csv
def save_active_generators(conventional, solar, wind):
    
    #conventional
    conventional_generator_array = np.array([conventional["nameplate"]/.95,conventional["year"],conventional["type"],conventional["efor"]])
    conventional_generator_df = pd.DataFrame(data=conventional_generator_array.T,
                                index=np.arange(conventional["nameplate"].size),
                                columns=["nameplate", "year", "type", "efor"])
    conventional_generator_df.to_csv("active_conventional.csv")

    #solar
    solar_generator_array = np.array([solar["nameplate"],solar["lat"],solar["lon"],solar["efor"]])

    solar_generator_df = pd.DataFrame(data=solar_generator_array.T,
                                index=np.arange(solar["nameplate"].size),
                                columns=["nameplate", "lat", "lon", "efor"])
    solar_generator_df.to_csv("active_solar.csv")

    #wind
    wind_generator_array = np.array([wind["nameplate"],wind["lat"],wind["lon"],wind["efor"]])

    wind_generator_df = pd.DataFrame(data=wind_generator_array.T,
                                index=np.arange(wind["nameplate"].size),
                                columns=["nameplate", "lat", "lon", "efor"])
    wind_generator_df.to_csv("active_wind.csv")
    return


##################### MAIN ##########################

def main(simulation,files,system,generator):

    global DEBUG 
    DEBUG = simulation["debug"]
    
    print('{:%Y-%m-%d %H:%M:%S}\tBegin Main'.format(datetime.datetime.now()))

    # get file data
    powGen_lats, powGen_lons, cf = get_powGen(files["solar cf file"],files["wind cf file"])
    hourly_load = get_demand_data(files["demand file"],simulation["year"],simulation["shift load"]) 
    
    
    # Load saved system
    if system["setting"] == "load":
        hourly_fleet_capacity = load_hourly_fleet_capacity(simulation,system,generator)

    # get fleet depending on input (option to load preprocessed saved fleet)
    else:
        #Get conventional system (nameplate capacity, year, technology, and efor)
        fleet_conventional_generators = get_conventional_fleet(files["eia folder"],system["region"],simulation["year"],
                                                                system["derate conventional"],system["conventional efor"])
        
        #Get RE system (nameplate capacity, latitude, longitude, and efor)
        fleet_solar_generators, fleet_wind_generators = get_solar_and_wind_fleet(files["eia folder"],system["region"],
                                                                                simulation["year"], system["RE efor"])

        # add lat/lon indices for cf matrix
        get_cf_index(fleet_solar_generators, powGen_lats, powGen_lons)
        get_cf_index(fleet_wind_generators, powGen_lats, powGen_lons)

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
        save_hourly_fleet_capacity(hourly_fleet_capacity,simulation,system,generator)
        

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
    

    ########## TESTING ##########
    np.savetxt('demand.csv',hourly_load,delimiter=',')
    np.savetxt('hourlyRisk.csv',hourlyRisk,delimiter=',')
    save_active_generators(fleet_conventional_generators,fleet_solar_generators,fleet_wind_generators)

    print('{:%Y-%m-%d %H:%M:%S}\tFinished Main'.format(datetime.datetime.now()))
    return elcc,hourlyRisk