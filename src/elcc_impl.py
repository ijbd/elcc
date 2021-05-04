import csv
from datetime import datetime, timedelta
import datetime
import math
import os
import sys
from os import path
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from numpy import random
from numpy import genfromtxt
import matplotlib
import matplotlib.pyplot as plt


np.random.seed()

# Globals
DEBUG = False
OUTPUT_DIRECTORY = ""

def get_powGen(solar_cf_file, wind_cf_file):
    
    """ Retrieve all necessary information from powGen netCDF files: RE capacity factors and corresponding lat/lons
    
    Capacity factors are in matrix of shape(lats, lon, 8760 hrs) for 1 year

    ...

    Args
    ----------
    `solar_cf_file` (str): file path to capacity factors of solar plants
        
    `wind_cf_file` (str): file path to capacity factors of wind plants
    """

    # Error Handling
    if not (path.exists(solar_cf_file) and path.exists(wind_cf_file)):
        error_message = 'Renewable Generation files not available:\n\t'+solar_cf_file+'\n\t'+wind_cf_file
        raise RuntimeError(error_message)
    
    solarPowGen = Dataset(solar_cf_file)
    windPowGen = Dataset(wind_cf_file) #assume solar and wind cover same geographic region

    powGen_lats = np.array(solarPowGen.variables['lat'][:])
    powGen_lons = np.array(solarPowGen.variables['lon'][:])

    cf = dict()
    cf["solar"] = np.array(solarPowGen.variables['cf'][:]) 
    cf["wind"] = np.array(windPowGen.variables['cf'][:])

    solarPowGen.close()
    windPowGen.close()

    return powGen_lats, powGen_lons, cf

def get_hourly_load(year,regions, hrsShift=0):
    """ Retrieve hourly load vector from load file

    ...

    Args:
    ----------
    `year` (int): year of interest

    `regions` (list): list of NERC Regions or Balancing Authorities to include

    `hrsShift` (int): optional parameter to shift load, default no shift
    """
    hourly_load = np.zeros(8760)

    for region in regions:

        load_file = "../demand/"+region+".csv"
        # error handling
        if not path.exists(load_file):
            error_message = "Invalid region or demand data unavailable. "+region
            raise RuntimeError(error_message)

        # Open file
        regional_load = pd.read_csv(load_file,delimiter=',',usecols=["date_time","cleaned demand (MW)"],index_col="date_time")

        # Remove leap days
        leap_days=regional_load.index[regional_load.index.str.find("-02-29",0,10) != -1]
        regional_load.drop(leap_days, inplace=True) 
            # two date_time formats from eia cleaned data
        leap_days=regional_load.index[regional_load.index.str.find(str(year)+"0229",0,10) != -1]
        regional_load.drop(leap_days, inplace=True)

        # Find Given Year
        hourly_regional_load = np.array(regional_load["cleaned demand (MW)"][regional_load.index.str.find(str(year),0,10) != -1].values)
        hourly_load += hourly_regional_load

    # Shift load
    if hrsShift!=0:
        newLoad = np.array([0]*abs(hrsShift))
        if hrsShift>0:
            hourly_load = np.concatenate([newLoad,hourly_load[:(hourly_load.size-hrsShift)]])
        else:
            hourly_load = np.concatenate([hourly_load[abs(hrsShift):],newLoad])

    print("Peak load:",np.amax(hourly_load),"MW")
    print('')
    return hourly_load

def get_total_interchange(year,regions,interchange_folder, hrsShift=0):
    """ Retrieve all imports/exports for a region during a given year

    ...

    Args:
    ----------
    `year` (int): year of interest

    `regions` (list): list of NERC Regions or Balancing Authorities to include

    `interchange_folder` (str): file path to folder containing total interchange data
    """
    total_interchange = np.zeros(8760)

    interchange_file_path = interchange_folder + "WECC_TI.csv"
    
    if not path.exists(interchange_file_path):
        error_message = "No interchange file found."
        raise RuntimeError(error_message)

    for region in regions:
    
        #loads in data from already cleaned total interchange data
        raw_TI_Data = pd.read_csv(interchange_file_path,usecols= ['UTC time',region],parse_dates= ['UTC time'])

        #selecting data for desired year, uses datetime format
        filtered_TI_data = raw_TI_Data[(raw_TI_Data['UTC time'].dt.year == year)]
        
        #cleaning for CISO, data is shifted forward 1 hour before 2016-9-13
        if ((region == "CISO") & (year == 2016)):
            #converting panda datetime to readable python date time
            datetime_array = raw_TI_Data['UTC time'].dt.to_pydatetime()

            #gets out time period of error
            ind = ((datetime_array >= pd.to_datetime('2016-01-01')) & (datetime_array <= pd.to_datetime('2016-09-13')))

            #shifts time period back 1 hour
            raw_TI_Data.loc[ind,region] = raw_TI_Data.loc[ 
            ind, region].shift(-1).values

            #selecting data for desired year, uses datetime format encoded in excel spreadsheet
            filtered_TI_data = raw_TI_Data[
                raw_TI_Data['UTC time'].dt.year == year
            ]

        #gets rid of any leap year day if applicable
        filtered_TI_data = filtered_TI_data[~((filtered_TI_data['UTC time'].dt.month == 2) & (filtered_TI_data['UTC time'].dt.day == 29))]
        
        #converting nan values to 0
        regional_interchange = filtered_TI_data[region].values
        regional_interchange[np.isnan(regional_interchange)] = 0
        
        if np.sum(regional_interchange) > 0:
            print(region+' '+str(year)+': Net Exporter')
        else:
            print(region+' '+str(year)+': Net Importer')

        total_interchange += regional_interchange
    
    # Shift interchange
    if hrsShift!=0:
        newInterchange = np.array([0]*abs(hrsShift))
        if hrsShift>0:
            total_interchange = np.concatenate([newInterchange,total_interchange[:-hrsShift]])
        else:
            total_interchange = np.concatenate([total_interchange[abs(hrsShift):],newInterchange])

    print('')

    return total_interchange

def get_temperature_data(temperature_file):
    """ Load hourly temperature (Celsius) for all the coordinates in desired region 

    ...

    Args:
    ----------
    `temperature_file` (str): file path to netCDF containing MERRA temperature data for entire region of interest (corresponding to powGen lats/lons)
    """

    temperature_data = np.array(Dataset(temperature_file)["T2M"][:][:][:]).T
    return (temperature_data-273.15)

def get_benchmark_fors(benchmark_FORs_file):
    """ Load in benchmark fors for temperature increments of 5 celsius from -15 to 35 for 6 different types of technology 

    ...

    Args:
    ----------
    `benchmark_FORs_file` (str): file path to excel file containing temperature dependent fors for different generator technologies
    """
    
    tech_categories = ["Temperature","HD","CC","CT","DS","HD","NU","ST","Other"]
    forData = pd.read_excel(benchmark_FORs_file)    
    benchmark_fors_tech = dict()
    for tech in tech_categories:
        benchmark_fors_tech[tech] = forData[tech].values
    return benchmark_fors_tech

def get_storage_fleet(eia_folder, region, year, round_trip_efficiency, efor, dispatch_strategy):
    """ Retrieve all active storage units

    ...

    Args:
    ----------
    `eia_folder` (string): file path to Form EIA-860 Folder

    `regions` (str): balancing authority

    `year` (int): year of interest

    `round_trip_efficiency` (float): efficiency for storage units
    
    `efor` (float): expected forced outage rate for storage units

    `dispatch strategy` (string): 'reliability' or 'arbitrage'... ARBITRAGE POORLY IMPLEMENTED
    """

    # Open files
    plants = pd.read_excel(eia_folder+"2___Plant_Y"+str(year)+".xlsx",skiprows=1,usecols=["Plant Code","NERC Region","Latitude",
                                                                                "Longitude","Balancing Authority Code"])
    all_storage_units = pd.read_excel(eia_folder+"3_4_Energy_Storage_Y"+str(year)+".xlsx",skiprows=1,\
                                                    usecols=["Plant Code","Technology","Nameplate Energy Capacity (MWh)","Status",
                                                            "Operating Year", "Maximum Charge Rate (MW)", "Maximum Discharge Rate (MW)"])
    
    # Filter by NERC Region and Balancing Authority to filter correct plant codes
    desired_plant_codes = plants["Plant Code"][plants["NERC Region"].isin(region) | plants["Balancing Authority Code"].isin(region)].values

    # Error Handling
    if desired_plant_codes.size == 0:
        error_message = "Invalid region(s): "+region
        raise RuntimeError(error_message)

    # filtering
    active_storage = all_storage_units[(all_storage_units["Plant Code"].isin(desired_plant_codes)) & (all_storage_units["Status"] == "OP")]
    active_storage = active_storage[active_storage["Nameplate Energy Capacity (MWh)"].astype(str) != " "]
    active_storage = active_storage[active_storage["Operating Year"] <= year]

    # get phs
    all_conventional_generators = pd.read_excel(eia_folder+"3_1_Generator_Y"+str(year)+".xlsx",skiprows=1,\
                                                    usecols=["Plant Code","Technology","Nameplate Capacity (MW)","Status",
                                                            "Operating Year"])

    # filter
    active_phs = all_conventional_generators[all_conventional_generators["Technology"]=='Hydroelectric Pumped Storage']
    active_phs = active_phs[active_phs["Plant Code"].isin(desired_plant_codes)]
    active_phs = active_phs[active_phs["Operating Year"] <= year]
    active_phs = active_phs[active_phs["Status"] == "OP"]

    # make phs dict
    phs = dict()
    phs["dispatch strategy"] = dispatch_strategy
    phs["num units"] = active_phs["Nameplate Capacity (MW)"].values.size
    phs["max charge rate"] = active_phs["Nameplate Capacity (MW)"].values
    phs["max discharge rate"] = active_phs["Nameplate Capacity (MW)"].values
    phs["max energy"] = active_phs["Nameplate Capacity (MW)"].values * 10. # assume 10 hour pumped hydro duration
    phs["roundtrip efficiency"] = np.ones(phs["num units"]) * .8 # from eia (average monthly efficiency)
    phs["one way efficiency"] = phs["roundtrip efficiency"] ** .5

    # make regular storage dict
    storage = dict()
    storage["dispatch strategy"] = dispatch_strategy
    storage["num units"] = active_storage["Nameplate Energy Capacity (MWh)"].values.size
    storage["max charge rate"] = active_storage["Maximum Charge Rate (MW)"].values
    storage["max discharge rate"] = active_storage["Maximum Discharge Rate (MW)"].values
    storage["max energy"] = active_storage["Nameplate Energy Capacity (MWh)"].values
    storage["roundtrip efficiency"] = np.ones(storage["num units"]) * round_trip_efficiency
    storage["one way efficiency"] = storage["roundtrip efficiency"] ** .5

    # combine phs
    storage = append_storage(storage, phs)

    # Hourly Tracking (storage starts empty)
    storage["power"] = np.zeros(storage["num units"])
    storage["extractable energy"] = np.ones(storage["num units"])*storage["max energy"]
    storage["energy"] = storage["extractable energy"] / storage["one way efficiency"]
    storage["time to discharge"] = storage["extractable energy"] / storage["max discharge rate"]
    storage["efor"] = efor
    storage["full"] = True

    sys.stdout.flush()

    return storage

def make_storage(include_storage, energy_capacity, charge_rate, discharge_rate, round_trip_efficiency, efor, dispatch_strategy):
# make a storage unit in properly formatted dictionary

    if include_storage == False or energy_capacity == 0:
        storage = dict()
        storage["num units"] = 0
        return storage

    storage = dict()

    storage["dispatch strategy"] = dispatch_strategy
    storage["num units"] = 1
    storage["max charge rate"] = np.array(charge_rate)
    storage["max discharge rate"] = np.array(discharge_rate)
    storage["max energy"] = np.array(energy_capacity)
    storage["roundtrip efficiency"] = np.ones(storage["num units"]) * round_trip_efficiency
    storage["one way efficiency"] = storage["roundtrip efficiency"] ** .5
    storage["power"] = np.zeros(storage["num units"])
    storage["extractable energy"] = np.ones(storage["num units"])*storage["max energy"]
    storage["energy"] = storage["extractable energy"] / storage["one way efficiency"]
    storage["time to discharge"] = storage["extractable energy"] / storage["max discharge rate"]
    storage["efor"] = efor
    storage["full"] = True

    return storage

def append_storage(fleet_storage, additional_storage):
    """ Combine two storage dictionaries
    ...
    Args:
        -----------
        `fleet_storage` (dict): dictionary of storage units
        `additional_storage` (dict): dictionary of storage units
    """
    #edge cases
    if fleet_storage["num units"] == 0:
        return additional_storage
    
    if additional_storage["num units"] == 0:
        return fleet_storage
    
    #combine regular attribute(s)
    fleet_storage["num units"] += additional_storage["num units"]
    
    #concatenate all array objects
    for key in fleet_storage:
        if isinstance(fleet_storage[key], np.ndarray):
            fleet_storage[key] = np.append(fleet_storage[key],additional_storage[key])
    
    return fleet_storage

def reset_storage(storage):
# All units begin full at the beginning of the year
    #for simulation begin empty
    storage["power"] = np.zeros(storage["num units"])
    storage["extractable energy"] = np.ones(storage["num units"])*storage["max energy"] # storage begins full
    storage["energy"] = storage["extractable energy"] / storage["one way efficiency"] 
    storage["time to discharge"] = storage["extractable energy"] / storage["max discharge rate"]
    storage["full"] = True
    
    return

def get_hourly_storage_contribution(num_iterations, hourly_capacity, hourly_load, storage, renewable_profile=None):
    """ Find the hourly capacity matrix for a set of storage units a given number of iterations.

        ...

        Args:
        ----------
        `num_iterations` (int): Number of capacity curves to sample for MCS.

        `hourly_capacity` (ndarray): array of hourly capacities for the desired number of iterations

        `hourly_load` (ndarray): vector of hourly load

        `storage` (dict): dictionary of storage units
        
        `renewable profile` (ndarray): vector of hourly renewable profiles
    """
    hourly_storage_contribution = np.zeros((8760,num_iterations))
    
    # edge case
    if storage["num units"] == 0:
        return 0

    # dispatch in every iteration according to outages and available capacity
    for i in range(num_iterations):

        if storage["dispatch strategy"] == "reliability":
            reliability_strategy(hourly_capacity[:,i],hourly_load,storage,hourly_storage_contribution[:,i])

        elif storage["dispatch strategy"] == "arbitrage":
            net_load = hourly_load - renewable_profile
            arbitrage_strategy(net_load,storage,hourly_storage_contribution[:,i])
        
        else:
            error_message = "Invalid dispatch strategy: \""+storage["dispatch strategy"]+"\""
            raise RuntimeWarning(error_message)
        reset_storage(storage)

    return hourly_storage_contribution

def arbitrage_strategy(net_load,storage,hourly_storage_contribution):
# BAD IMPLEMENTATION : emulate arbitrage for storage dispatch policy
    for day in range(365):
        start = day*24
        end = (day+1)*24
        arbitrage_dispatch(net_load[start:end],storage,hourly_storage_contribution[start:end])
    
    return 

def arbitrage_dispatch(net_load, storage, hourly_storage_contribution):
# BAD IMPLEMENTATION : shoddy arbitrage emulation with percentile-based peak shaving
    discharge_percentile = 75
    charge_percentile = 25

    discharge_threshold = np.percentile(net_load, discharge_percentile)
    charge_threshold = np.percentile(net_load, charge_percentile)

    for hour in range(24):
        if net_load[hour] > discharge_threshold:
            load_difference = net_load[hour] - discharge_threshold
            hourly_storage_contribution[hour] = discharge_storage(load_difference, storage)
        elif net_load[hour] < charge_threshold:
            load_difference = charge_threshold - net_load[hour]
            hourly_storage_contribution[hour] = charge_storage(load_difference, storage)

    return

def reliability_strategy(hourly_capacity,hourly_load,storage,hourly_storage_contribution):
# Charge/Discharge storage to greedily maximize reliability    
    simulation_days = np.unique((np.argwhere(hourly_load > hourly_capacity)//24).flatten())
    simulation_days = np.unique(np.minimum(np.maximum(simulation_days,0),364))
    
    # no risk days
    if len(simulation_days) != 0:
        last_day = simulation_days[-1]
        
    #choose strategy on a daily basis
    for i in range(simulation_days.size):
        day = simulation_days[i]
        if day == last_day:
            next_risk_day = day + 1
        else:
            next_risk_day = simulation_days[i+1]
        # Simulate risk days or until storage is charged
        storage["full"] = False
        while day != next_risk_day  and storage["full"] == False:
            start = (day)*24 
            end = (day+1)*24
            reliability_dispatch(hourly_storage_contribution[start:end], hourly_load[start:end],
                                hourly_capacity[start:end], storage)
            day += 1
    
    return

def reliability_dispatch(hourly_storage_contribution, hourly_load, hourly_capacity, storage):
# Disharge when load exceeds capacity, charge otherwise
    floating_point_buffer = 1e-6 # 1 W buffer to avoid false loss-of-loads

    for hour in range(24):
        # discharge if load is not met
        if hourly_load[hour] > hourly_capacity[hour]:
            unmet_load = hourly_load[hour] - hourly_capacity[hour] + floating_point_buffer
            # discharge storage
            hourly_storage_contribution[hour] = discharge_storage(unmet_load, storage)
        # charge if surplus
        else:
            additional_capacity = hourly_capacity[hour] - hourly_load[hour] - floating_point_buffer
            # charge storage
            hourly_storage_contribution[hour] = charge_storage(additional_capacity, storage)

    return

def discharge_storage(unmet_load, storage):
# Discharge according to optimal policy proposed by Evans et.al.
    P_r = unmet_load
    p = storage["max discharge rate"]
    x = storage["time to discharge"]
    u = storage["power"]

    # sort unique "time to discharges", discharge HIGHEST first
    y = np.flip(np.unique(np.concatenate((x,np.maximum(x-1,0)))))
    E_max = 0
    i = 0

    #Find devices partaking in discharge
    while E_max < P_r and i < len(y)-1:
        i += 1
        E_min = E_max
        E_max = np.sum(p*np.maximum(np.minimum(x-y[i],1),0))

    #Load exactly met or unmeetable load
    if E_max <= P_r:
        z = y[i]

    #interpolate
    else:
        z = y[i-1] + (P_r - E_min)/(E_max - E_min)*(y[i]-y[i-1])
        
    
    u = p*np.maximum(np.minimum(x-z,1),0) * (np.random.random_sample(x.shape)>storage["efor"])

    if storage["efor"] > 0:
        u *= np.random.random_sample(x.shape)>storage["efor"]

    #update storage 
    storage["power"] = u
    update_storage(storage,"discharge")

    return np.sum(storage["power"])

def charge_storage(additional_capacity, storage):
# Charge according to policy proposed by Evans et.al. 
    P_r = -additional_capacity
    p_c = -storage["max charge rate"]
    p_d = storage["max discharge rate"]
    x_max = storage["max energy"] / storage["max discharge rate"]
    x = storage["time to discharge"]
    
    u = storage["power"]

    n = storage["roundtrip efficiency"]
    
    # sort unique "time to discharges", charge LOWEST first
    z_max = np.minimum(x-n*p_c/p_d, x_max)
    y = np.unique(np.concatenate((x,z_max)))

    E_max = 0
    E_min = 0
    i = 0

    # Find devices partaking in charge
    while E_max < -P_r and i < len(y)-1:
        i += 1
        E_min = E_max
        E_max = np.sum(p_d/n*np.maximum(np.minimum(y[i],z_max)-x,0))

    if E_max <= -P_r:
        z = y[i]
    #interpolate
    else:
        if E_max == 0 and E_min == 0:
            return 0
        else:
            z = y[i-1] + (-P_r - E_min)/(E_max - E_min)*(y[i]-y[i-1])
        
    
    u = -1*p_d/n*np.maximum(np.minimum(z,z_max)-x,0)

    if storage["efor"] > 0:
        u *= np.random.random_sample(x.shape)>storage["efor"]
        
    #update storage 
    storage["power"] = u
    update_storage(storage, "charge")

    return np.sum(storage["power"])

def update_storage(storage, status):
# Update charge status of all storage devices

    if status == "discharge":
        storage["extractable energy"] = storage["extractable energy"] - storage["power"] 
        storage["energy"] = storage["extractable energy"] / storage["one way efficiency"]
    if status == "charge":
        storage["energy"] = storage["energy"] - storage["power"] * storage["one way efficiency"] 
        storage["extractable energy"] = storage["energy"] * storage["one way efficiency"]
    
    # set storage state
    storage["full"] = np.sum(storage["extractable energy"]) == np.sum(storage["max energy"])

    storage["time to discharge"] = np.divide(storage["extractable energy"],storage["max discharge rate"]) 

    return

def calculate_fors(total_efor_array, simplified_tech_list, benchmark_fors,hourly_temp_data):  
# Compute the FOR given an input of temperatures and a specific technology type

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
    
    #TESTING
    #print("Average annual temperature dependent FOR: " + str(np.average(total_efor_array)))
    
    return total_efor_array

def get_tech_efor_round_downs(simplified_tech_list, latitudes, longitudes,temperature_data,benchmark_fors):
# Create total forced outage rate for all the generators in desired region's fleet    
    total_efor_array = np.zeros((len(simplified_tech_list),8760))
    hourly_temp_data = temperature_data[longitudes,latitudes]

    simplified_tech_list = np.array([simplified_tech_list,]*8760).T
    
    return calculate_fors(total_efor_array, simplified_tech_list, benchmark_fors, hourly_temp_data)

def find_desired_tech_indices(desired_tech_list,generator_technology):
# Function used to convert technologies of all generators into 6 known for technology relationships
    simplified_tech_list = np.zeros(len(generator_technology))
    generator_technology = pd.DataFrame(data=generator_technology.flatten())
    for tech_type in desired_tech_list:
        specific_tech = generator_technology.isin(desired_tech_list[tech_type])
        simplified_tech_list = np.where((generator_technology[specific_tech].fillna(0).values).flatten() != 0,tech_type,simplified_tech_list) 
    return simplified_tech_list

def get_temperature_dependent_efor(latitudes,longitudes,technology,temperature_data,benchmark_fors):
# Create main tech list where all the other different types of tech are divided into 6 main known temperature-FOR relatonship 
    total_tech_list = dict()
    total_tech_list["CC"] = np.array(["Natural Gas Fired Combined Cycle"])
    total_tech_list["CT"] = np.array(["Natural Gas Fired Combustion Turbine","Landfill Gas",])
    total_tech_list["DS"] = np.array(["Natural Gas Internal Combustion Engine"])
    total_tech_list["ST"]  = np.array(["Conventional Steam Coal","Natural Gas Steam Turbine"])
    total_tech_list["NU"]  = np.array(["Nuclear"])
    total_tech_list["HD"]  =  np.array(["Conventional Hydroelectric","Solar Thermal without Energy Storage",
                   "Hydroelectric Pumped Storage","Solar Thermal with Energy Storage","Wood/Wood Waste Biomass"])
    simplified_tech_list = find_desired_tech_indices(total_tech_list,technology)

    return get_tech_efor_round_downs(simplified_tech_list,latitudes,longitudes,temperature_data,benchmark_fors)

def get_conventional_fleet_impl(plants,active_generators,system_preferences,temperature_data, year,powGen_lats,powGen_lons,benchmark_fors):
# Filter generators for operation status, technology, year of simulation

    # filtering
    active_generators = active_generators[(active_generators["Operating Year"] <= year)]
    active_generators = active_generators[(active_generators["Status"] == "OP")]

    not_conventional = ['Solar Photovoltaic','Onshore Wind Turbine','Offshore Wind Turbine','Batteries','Hydroelectric Pumped Storage']
    active_generators = active_generators[(~active_generators["Technology"].isin(not_conventional))]

    ### DBG
    print('DBG ERASE... (EXITING)')
    print(np.unique(active_generators['Technology'].values))
    sys.exit(0)
    
    # Fill empty summer/winter capacities
    active_generators["Summer Capacity (MW)"].where(active_generators["Summer Capacity (MW)"].astype(str) != " ",
                                                    active_generators["Nameplate Capacity (MW)"], inplace=True)
    active_generators["Winter Capacity (MW)"].where(active_generators["Winter Capacity (MW)"].astype(str) != " ", 
                                                    active_generators["Nameplate Capacity (MW)"], inplace=True)
    
    #getting lats and longs correct indices
    plants.set_index("Plant Code",inplace=True)
    latitudes = find_nearest_impl(plants["Latitude"][active_generators["Plant Code"]].values,powGen_lats)
    longitudes = find_nearest_impl(plants["Longitude"][active_generators["Plant Code"]].values,powGen_lons)

    # Convert Dataframe to Dictionary of numpy arrays
    conventional_generators = dict()
    conventional_generators["num units"] = active_generators["Nameplate Capacity (MW)"].values.size
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
    
    return conventional_generators

def get_conventional_fleet(eia_folder, regions, year, system_preferences,powGen_lats,powGen_lons,temperature_data,benchmark_fors):
    """ Retrieve all active conventional generators

    ...

    Args:
    ----------
    `eia_folder` (string): file path to Form EIA-860 Folder

    `regions` (str): balancing authority

    `year` (int): year of interest

    `system_preferences` (dict): set of parameters. Must include keys 'temperature dependent FOR', 'temperature dependent FOR indpendent of size', and 'conventional efor'
    
    `powGen_lats` (ndarray): vector of latitudes to map temperature data to

    `powGen_lons` (ndarray): vector of longitudes to map temperature data to
    
    `temperature_data` (ndarray): array of temperatures in Celsius to map generators to. Dimensions correspond to 'powGen_lats' and 'powGen_lons'

    `benchmark_fors` (ndarray): array of temperature-dependent FORs for different generator technologies

    """

    # Open files
    plants = pd.read_excel(eia_folder+"2___Plant_Y"+str(year)+".xlsx",skiprows=1,usecols=["Plant Code","NERC Region","Latitude",
                                                                                "Longitude","Balancing Authority Code"])
    all_conventional_generators = pd.read_excel(eia_folder+"3_1_Generator_Y"+str(year)+".xlsx",skiprows=1,\
                                                    usecols=["Plant Code","Generator ID","Technology","Nameplate Capacity (MW)","Status",
                                                            "Operating Year", "Summer Capacity (MW)", "Winter Capacity (MW)"])
    # Filter by NERC Region and Balancing Authority to filter correct plant codes
    desired_plant_codes = plants["Plant Code"][plants["NERC Region"].isin(regions) | plants["Balancing Authority Code"].isin(regions)].values

    # Error Handling
    if desired_plant_codes.size == 0:
        error_message = "Invalid region(s): " + str(regions)
        raise RuntimeError(error_message)

    # Get operating generators
    active_generators = all_conventional_generators[(all_conventional_generators["Plant Code"].isin(desired_plant_codes))]

    # Get partially-owned plants
    active_generators = add_partial_ownership_generators(eia_folder, regions, year, active_generators, all_conventional_generators,True)
    
    return get_conventional_fleet_impl(plants,active_generators,system_preferences,temperature_data,year,powGen_lats,powGen_lons,benchmark_fors)
    
def get_RE_fleet_impl(eia_folder, regions, year, plants, RE_generators, desired_plant_codes, RE_efor, renewable_multiplier):
# Filter generators for operation status, and map generators to powGen coordinates

    # Get generators in region
    active_generators = RE_generators[(RE_generators["Plant Code"].isin(desired_plant_codes))]

    # Get partially-owned plants
    active_generators = add_partial_ownership_generators(eia_folder, regions, year, active_generators, RE_generators)

    # filtering
    active_generators = active_generators[active_generators["Status"] == 'OP']
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
    RE_generators["num units"] = active_generators["Nameplate Capacity (MW)"].values.size 
    RE_generators["nameplate"] = active_generators["Nameplate Capacity (MW)"].values * renewable_multiplier
    RE_generators["summer nameplate"] = active_generators["Summer Capacity (MW)"].values * renewable_multiplier
    RE_generators["winter nameplate"] = active_generators["Winter Capacity (MW)"].values * renewable_multiplier
    RE_generators["lat"] = latitudes
    RE_generators["lon"] = longitudes
    RE_generators["efor"] = np.ones(RE_generators["nameplate"].size) * RE_efor 

    return RE_generators

def get_solar_and_wind_fleet(eia_folder, regions, year, RE_efor, powGen_lats, powGen_lons, renewable_multiplier):
    """ Retrieve all active wind and solar generators

    ...

    Args:
    ----------
    `eia_folder` (string): file path to Form EIA-860 Folder

    `regions` (list): list of NERC Regions or Balancing Authorities to include

    `year` (int): year of interest

    `RE_efor` (float): Forced outage rate for solar and wind generators
    
    `powGen_lats` (ndarray): vector of latitudes corresponding to capacity factor array

    `powGen_lons` (ndarray): vector of longitudes corresponding to capacity factor array
    """

    # Open files
    plants = pd.read_excel(eia_folder+"2___Plant_Y"+str(year)+".xlsx",skiprows=1,usecols=[  "Plant Code","NERC Region","Latitude",
                                                                                            "Longitude","Balancing Authority Code"])
    all_solar_generators = pd.read_excel(eia_folder+"3_3_Solar_Y"+str(year)+".xlsx",skiprows=1,\
                                usecols=["Plant Code","Generator ID","Nameplate Capacity (MW)",
                                        "Summer Capacity (MW)", "Winter Capacity (MW)",
                                        "Status","Operating Year"])
    all_wind_generators = pd.read_excel(eia_folder+"3_2_Wind_Y"+str(year)+".xlsx",skiprows=1,\
                                usecols=["Plant Code","Generator ID","Nameplate Capacity (MW)",
                                        "Summer Capacity (MW)", "Winter Capacity (MW)",
                                        "Status","Operating Year"])

     # Sort by NERC Region and Balancing Authority to filter correct plant codes
    nerc_region_plant_codes = plants["Plant Code"][plants["NERC Region"].isin(regions)].values
    balancing_authority_plant_codes = plants["Plant Code"][plants["Balancing Authority Code"].isin(regions)].values
    
    desired_plant_codes = np.concatenate((nerc_region_plant_codes, balancing_authority_plant_codes))

    # Repeat process for solar and wind
    plants.set_index("Plant Code",inplace=True)
    solar_generators = get_RE_fleet_impl(eia_folder,regions,year,plants,all_solar_generators,desired_plant_codes,RE_efor,renewable_multiplier)
    wind_generators = get_RE_fleet_impl(eia_folder,regions,year,plants,all_wind_generators,desired_plant_codes,RE_efor,renewable_multiplier)

    solar_generators["generator type"] = "solar"
    wind_generators["generator type"] = "wind"

    # Process for lat,lon indices
    solar_generators = get_cf_index(solar_generators,powGen_lats,powGen_lons)
    wind_generators = get_cf_index(wind_generators,powGen_lats,powGen_lons)

    return solar_generators, wind_generators

def add_partial_ownership_generators(eia_folder,regions,year,generators,all_generators,print_utilities=False):
# Add partially owned generators. !!! Obsolete until dictionary is filled with balancing authorities and associated utilities

    # Working dictionary for utilities associated with balancing authorities        
    known_utilities = dict()

    utilities = np.array([known_utilities[region] for region in regions if region in known_utilities]).flatten()

    if len(utilities) != 0:
        if print_utilities:
            print('Utilites:', utilities)
            print('')
    else:
        return generators

    # EIA 860 schedule 4
    owners = pd.read_excel(eia_folder+"4___Owner_Y"+str(year)+".xlsx",skiprows=1,usecols=[  "Plant Code","Generator ID",
                                                                                            "Status","Owner Name","Percent Owned"])

    # filtering
    owners = owners[owners["Owner Name"].isin(utilities)]

    if len(owners["Plant Code"]) == 0 and print_utilities:
            print('No partially-owned plants found')
    
    generators = generators[~(generators["Plant Code"].isin(owners["Plant Code"]))]

    if owners.empty:
        return generators
    
    total_added = 0

    for ind, row in owners.iterrows():
            generator = all_generators[     (all_generators["Plant Code"] == row["Plant Code"]) &\
                                            (all_generators["Generator ID"] == row["Generator ID"])]
            if generator.empty:
                continue
            partial_generator = generator.copy()
            idx = partial_generator.index[0]                       
            partial_generator.at[idx,"Nameplate Capacity (MW)"] *= row["Percent Owned"]
            partial_generator.at[idx,"Summer Capacity (MW)"] *= row["Percent Owned"]
            partial_generator.at[idx,"Winter Capacity (MW)"] *= row["Percent Owned"]
            generators = generators.append(partial_generator)

            total_added += partial_generator.at[idx,"Nameplate Capacity (MW)"]
    
    return generators

def find_nearest_impl(actual_coordinates, discrete_coordinates):
# Find index of nearest coordinate for vector of coordinates   
    indices = []
    for coord in actual_coordinates:
        indices.append((np.abs(coord-discrete_coordinates)).argmin())
    return np.array(indices)

def get_cf_index(RE_generators, powGen_lats, powGen_lons):
# Convert the latitudes and longitudes of the vg into indices for capacity factor matrix
    RE_generators["lat idx"] = find_nearest_impl(RE_generators["lat"], powGen_lats).astype(int)
    RE_generators["lon idx"] = find_nearest_impl(RE_generators["lon"], powGen_lons).astype(int)

    return RE_generators

def get_hourly_RE_impl(RE_generators, cf):
# Find expected hourly capacity for RE generators before sampling outages. Of shape (8760 hrs, num generators)
    
    # combine summer and winter capacities
    RE_winter_nameplate = np.tile(RE_generators["winter nameplate"],(8760//4,1))
    RE_summer_nameplate = np.tile(RE_generators["summer nameplate"],(8760//2,1))
    RE_nameplate = np.vstack((RE_winter_nameplate,RE_summer_nameplate,RE_winter_nameplate))

    # multiply by variable hourly capacity factor
    hours = np.tile(np.arange(8760),(RE_generators["nameplate"].size,1)).T # shape(8760 hrs, num generators)
    RE_capacity = np.multiply(RE_nameplate, cf[RE_generators["lat idx"], RE_generators["lon idx"], hours])
    return RE_capacity

def get_RE_profile_for_storage(cf, *generators):
# Return expected renewable output for generator. Used for storage
    renewable_profile = np.zeros(8760)
    for generator in generators:
        renewable_profile = np.add(renewable_profile,np.sum(get_hourly_RE_impl(generator,cf[generator["generator type"]]),axis=1))
    return renewable_profile

def sample_outages_impl(num_iterations, pre_outage_capacity, generators):
# Get hourly capacity matrix for a generator by sampling outage rates over all hours/iterations. Of shape (8760 hrs, num iterations)
    hourly_capacity = np.zeros((8760,num_iterations))
    # otherwise sample outages and add generator contribution
    max_iterations = 2000 // generators["nameplate"].size # the largest # of iterations to compute at one time (solve memory issues)
    if max_iterations == 0: 
        max_iterations = 1
    for i in range(num_iterations // max_iterations):
        for_matrix = np.random.random_sample((max_iterations,8760,generators["nameplate"].size))>(generators["efor"].T) # shape(its,hours,generators)
        capacity = np.sum(np.multiply(pre_outage_capacity,for_matrix),axis=2).T # shape(its,hours).T -> shape(hours,its)
        hourly_capacity[:,i*max_iterations:(i+1)*max_iterations] = capacity 
    if num_iterations % max_iterations != 0:
        remaining_iterations = num_iterations % max_iterations
        for_matrix = np.random.random_sample((remaining_iterations,8760,generators["nameplate"].size))>(generators["efor"].T)
        capacity = np.sum(np.multiply(pre_outage_capacity,for_matrix),axis=2).T
        hourly_capacity[:,-remaining_iterations:] = capacity
    return hourly_capacity

def get_hourly_capacity(num_iterations, generators, cf=None):
    """ Find the hourly capacity matrix for a set of generators for a given number of iterations.

        ...

        Args:
        ----------
        `num_iterations` (int): Number of capacity curves to sample for MCS.

        `generators` (dict): dictionary of generators. Must contain keys 'nameplate', 'summer nameplate', 'winter nameplate', and 'efor'

        `year` (int): year of interest

        `RE_efor` (float): Forced outage rate for solar and wind generators
        
        `powGen_lats` (ndarray): vector of latitudes corresponding to capacity factor array

        `powGen_lons` (ndarray): vector of longitudes corresponding to capacity factor array
    """

    if generators["num units"] == 0:
        return 0

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

def get_hourly_fleet_capacity(num_iterations, conventional_generators, solar_generators, wind_generators, cf, storage_units=None, hourly_load=None, renewable_profile=None):
    """ Find the hourly capacity matrix for the entire fleet for a given number of iterations.

        ...

        Args:
        ----------
        `num_iterations` (int): Number of capacity curves to sample for MCS.

        `conventional_generators` (dict): dictionary of conventional generators. Must contain keys 'nameplate', 'summer nameplate', 'winter nameplate', and 'efor'

        `solar_generators` (dict): dictionary of solar generators. Must contain keys 'nameplate', 'summer nameplate', 'winter nameplate', 'lat idx', and 'lon idx'

        `wind_generators` (dict): dictionary of wind generators. Must contain keys 'nameplate', 'summer nameplate', 'winter nameplate', 'lat idx', and 'lon idx'

        `cf` (dict): dictionary containing two numpy arrays for solar and wind capacity factors

        `storage_units` (dict): OPTIONAL dictionary of storage units. IF INCLUDED: must include hourly_load and renewable_profile
        
        `hourly_load` (ndarray): OPTIONAL vector of hourly load. Only include with storage_units

        `renewable_profile` (ndarray): OPTIONAL vector of hourly renewable output. Only include with storage_units
    """

    hourly_fleet_capacity = np.zeros((8760,num_iterations))

    # conventional, solar, and wind
    hourly_fleet_capacity += get_hourly_capacity(num_iterations,conventional_generators)
    hourly_fleet_capacity += get_hourly_capacity(num_iterations,solar_generators,cf["solar"])
    hourly_fleet_capacity += get_hourly_capacity(num_iterations,wind_generators,cf["wind"])

    if storage_units is not None:
        hourly_fleet_capacity += get_hourly_storage_contribution(   num_iterations,hourly_fleet_capacity,
                                                                    hourly_load,storage_units,renewable_profile)
    
    return hourly_fleet_capacity

def get_lolh(num_iterations, hourly_capacity, hourly_load):
# Calculate number of expected hours in which load does not meet demand using monte carlo method

    # identify where load exceeds capacity (loss-of-load). Of shape(8760 hrs, num iterations)
    lol_matrix = np.where(hourly_load > hourly_capacity.T, 1, 0).T
    hourly_risk = np.sum(lol_matrix,axis=1) / float(num_iterations)
    lolh = np.sum(hourly_risk)

    return lolh, hourly_risk

def remove_oldest_impl(generators, manual_oldest_year=0):
# Remove the oldest thermal generators from the conventional system. After all thermal generators are removed, remove the oldest hydro generators.

    # non-hydro plants
    not_hydro = generators["technology"] != "Conventional Hydroelectric"

    # while conventional units still exist, remove those first
    if len(generators["nameplate"][not_hydro]) != 0:
        

        # find oldest plant
        oldest_year = np.amin(generators["year"][not_hydro]) 

        # check for manual removal
        if manual_oldest_year > oldest_year:
            oldest_year = manual_oldest_year

        # erase all generators older than that year
        erase = np.logical_and(generators["year"] <= oldest_year, not_hydro)

    # if there are no more conventional units, remove hydro
    else:
        # find smallest plant
        smallest_capacity = np.amin(generators["nameplate"])

        # erase all generators with capacities smaller than this
        erase = generators["nameplate"] <= smallest_capacity

        # avoid error
        oldest_year = 9999

    capacity_removed = np.sum(generators["nameplate"][erase])

    generators["nameplate"] = generators["nameplate"][np.logical_not(erase)]
    generators["summer nameplate"] = generators["summer nameplate"][np.logical_not(erase)]
    generators["winter nameplate"] = generators["winter nameplate"][np.logical_not(erase)]
    generators["year"] = generators["year"][np.logical_not(erase)]
    generators["technology"] = generators["technology"][np.logical_not(erase)]
    generators["efor"] = generators["efor"][np.logical_not(erase)]

    generators["num units"] = len(generators["nameplate"])

    return generators, oldest_year, capacity_removed

def remove_generator_binary_constraints(lolh, target_lolh, generator_size_max, generator_size_min, generator_size):
# Constraints for the 'remove_generators' binary search    
    convergence_not_met = generator_size_max - generator_size_min > 2
    reliabilility_not_met = abs(target_lolh - lolh) > 1e-9
    not_zero_met = generator_size != 0

    return convergence_not_met and reliabilility_not_met and not_zero_met

def remove_generators(  num_iterations, conventional_generators, solar_generators, wind_generators, storage_units, cf, 
                        hourly_load, oldest_year_manual, target_lolh, temperature_dependent_efor, conventional_efor, renewable_profile):
    """ Remove generators to meet reliability target.

    ...

    Args:
        ----------
        `num_iterations` (int): Number of capacity curves to sample for MCS.

        `conventional_generators` (dict): dictionary of conventional generators. Must contain keys 'nameplate', 'summer nameplate', 'winter nameplate', and 'efor'

        `solar_generators` (dict): dictionary of solar generators. Must contain keys 'nameplate', 'summer nameplate', 'winter nameplate', 'lat idx', and 'lon idx'

        `wind_generators` (dict): dictionary of wind generators. Must contain keys 'nameplate', 'summer nameplate', 'winter nameplate', 'lat idx', and 'lon idx'

        `storage_units` (dict): dictionary of storage units. 
        
        `cf` (dict): dictionary containing two numpy arrays for solar and wind capacity factors

        `hourly_load` (ndarray): vector of hourly load. 

        `oldest_year_manual` (int): year before which thermal generators will manually be removed 

        `target_LOLH` (float): reliability target

        `temperature_dependent_efor` (bool): indicates whether or not to temperature dependent forced outage rates were used.

        `conventional_efor` (float): forced outage rate to use for the supplemental generators

        `renewable_profile` (ndarray): vector of hourly renewable output

    """

    precision = int(math.log10(num_iterations))

    # Remove capacity until reliability drops beyond target LOLH/year (low iterations to save time)
 
    low_iterations = min(50,num_iterations)
    total_capacity_removed = 0
    oldest_year = np.amin(conventional_generators["year"][conventional_generators["technology"] != "Conventional Hydroelectric"]) 
    
    # manual removal
    if oldest_year_manual > oldest_year:
        conventional_generators, oldest_year, capacity_removed = remove_oldest_impl(conventional_generators, oldest_year_manual)
        total_capacity_removed += capacity_removed 

    # Find original reliability
    hourly_fleet_capacity = get_hourly_fleet_capacity(low_iterations,conventional_generators,solar_generators,
                                                        wind_generators,cf,storage_units,hourly_load,renewable_profile)
    lolh, hourly_risk = get_lolh(low_iterations,hourly_fleet_capacity,hourly_load) 
    
    # Error Handling: Under Reliable System
    if lolh >= target_lolh:
        print("LOLH:", round(lolh,2))
        print("LOLH already greater than target. Under reliable system.")

        if DEBUG:
            print("Hour of year:",np.argwhere(hourly_risk != 0).flatten())
            print("Hour of Day:",np.argwhere(hourly_risk != 0).flatten()%24)
            print("Risk:",hourly_risk[hourly_risk != 0].flatten()*50)
            np.savetxt(OUTPUT_DIRECTORY+'hourly_risk.csv',hourly_risk,delimiter=',')
        

    while conventional_generators["nameplate"].size > 1 and lolh < target_lolh:
        conventional_generators, oldest_year, capacity_removed = remove_oldest_impl(conventional_generators)
        hourly_fleet_capacity = get_hourly_fleet_capacity(low_iterations,conventional_generators,solar_generators,
                                                            wind_generators,cf,storage_units,hourly_load,renewable_profile)
        lolh, hourly_risk = get_lolh(low_iterations,hourly_fleet_capacity,hourly_load) 
        total_capacity_removed += capacity_removed

        print("Oldest Year:\t",int(oldest_year),"\tLOLH:\t",round(lolh,2),"\tCapacity Removed:\t",capacity_removed)
    
    print('')

    # find reliability of higher iteration simulation

    hourly_fleet_capacity = get_hourly_fleet_capacity(num_iterations,conventional_generators,solar_generators,
                                                        wind_generators,cf)

    hourly_storage_capacity = get_hourly_storage_contribution(  num_iterations,
                                                                hourly_fleet_capacity, 
                                                                hourly_load, 
                                                                storage_units,
                                                                renewable_profile)
                                                                
    hourly_total_capacity = hourly_fleet_capacity + hourly_storage_capacity 

    lolh, hourly_risk = get_lolh(num_iterations, hourly_total_capacity, hourly_load)

    # bad sample remove more generators
    if lolh < target_lolh:

        low_iterations *= 5

        while conventional_generators["nameplate"].size > 1 and lolh < target_lolh:
            
            conventional_generators, oldest_year, capacity_removed = remove_oldest_impl(conventional_generators)
            hourly_fleet_capacity = get_hourly_fleet_capacity(low_iterations,conventional_generators,solar_generators,
                                                                wind_generators,cf,storage_units,hourly_load,renewable_profile)
            lolh, hourly_risk = get_lolh(low_iterations,hourly_fleet_capacity,hourly_load) 
            total_capacity_removed += capacity_removed
            print("Oldest Year:\t",int(oldest_year),"\tLOLH:\t",round(lolh,2),"\tCapacity Removed:\t",capacity_removed,flush=True)
        
        hourly_fleet_capacity = get_hourly_fleet_capacity(num_iterations,conventional_generators,solar_generators,
                                                        wind_generators,cf)

        hourly_storage_capacity = get_hourly_storage_contribution(  num_iterations,
                                                                    hourly_fleet_capacity, 
                                                                    hourly_load, 
                                                                    storage_units,
                                                                    renewable_profile)
                                                                    
        hourly_total_capacity = hourly_fleet_capacity + hourly_storage_capacity 

        lolh, hourly_risk = get_lolh(num_iterations, hourly_total_capacity, hourly_load)

    # add supplemental units to match target reliability

    supplemental_capacity = 0
    supplemental_generator_unit_size = 50
    hourly_supplemental_unit_capacity = 0 # add one unit at a time to adjust generator size if necessary

    # add supplemental generators of constant size until system is over reliable
    while lolh > target_lolh:

        # make new generator
        supplemental_generator = make_conventional_generator(supplemental_generator_unit_size, 
                                                            conventional_efor, temperature_dependent_efor)

        hourly_supplemental_unit_capacity = get_hourly_capacity( num_iterations, supplemental_generator)
        

        # find new reliability
        hourly_storage_capacity = get_hourly_storage_contribution(  num_iterations,
                                                                    hourly_fleet_capacity+hourly_supplemental_unit_capacity, 
                                                                    hourly_load, 
                                                                    storage_units,
                                                                    renewable_profile)
        hourly_total_capacity = hourly_fleet_capacity + hourly_supplemental_unit_capacity + hourly_storage_capacity

        lolh, hourly_risk = get_lolh(num_iterations,hourly_total_capacity,hourly_load)        
        
        # add supplemental capacity fleet in increments
        if lolh > target_lolh:
            supplemental_capacity += supplemental_generator_unit_size
            hourly_fleet_capacity += hourly_supplemental_unit_capacity
            print("Supplement Capacity:\t",int(supplemental_capacity),"\tLOLH:\t", round(lolh,precision),flush=True)
    
    #binary search to find last supplemental generator size

    generator_size_max = supplemental_generator_unit_size
    generator_size_min = 0
    generator_size_old = supplemental_generator_unit_size
    generator_size_new = generator_size_max / 2

    hourly_supplemental_unit_capacity = hourly_supplemental_unit_capacity / generator_size_old * generator_size_new

    hourly_storage_capacity = get_hourly_storage_contribution(  num_iterations,
                                                                hourly_fleet_capacity+hourly_supplemental_unit_capacity, 
                                                                hourly_load, 
                                                                storage_units,
                                                                renewable_profile)

    hourly_total_capacity = hourly_fleet_capacity + hourly_supplemental_unit_capacity + hourly_storage_capacity

    lolh, hourly_risk = get_lolh(num_iterations,hourly_total_capacity,hourly_load)  

    print("Supplement Capacity:\t",int(supplemental_capacity+generator_size_new),"\tLOLH:\t", round(lolh,precision))

    while remove_generator_binary_constraints(lolh, target_lolh, generator_size_max, generator_size_min, generator_size_new):

        generator_size_old = generator_size_new

        if lolh > target_lolh: #under reliable
            generator_size_min = generator_size_new
            generator_size_new = int((generator_size_min + generator_size_max)/2)
        else: #over reliable
            generator_size_max = generator_size_new
            generator_size_new = int((generator_size_min + generator_size_max)/2)

        # find new reliability
        hourly_supplemental_unit_capacity = hourly_supplemental_unit_capacity / generator_size_old * generator_size_new

        hourly_storage_capacity = get_hourly_storage_contribution(  num_iterations,
                                                                    hourly_fleet_capacity+hourly_supplemental_unit_capacity, 
                                                                    hourly_load, 
                                                                    storage_units,
                                                                    renewable_profile)

        hourly_total_capacity = hourly_fleet_capacity + hourly_supplemental_unit_capacity + hourly_storage_capacity

        lolh, hourly_risk = get_lolh(num_iterations,hourly_total_capacity,hourly_load)  

        print("Supplement Capacity:\t",int(supplemental_capacity+generator_size_new),"\tLOLH:\t", round(lolh,precision),flush=True)
    
    print('')

    # add supplemental generators to fleet

    supplemental_capacity += generator_size_new
    hourly_fleet_capacity += hourly_supplemental_unit_capacity

    supplemental_generators = make_supplemental_generators( supplemental_capacity, conventional_efor, 
                                                            temperature_dependent_efor, supplemental_generator_unit_size)
    conventional_generators = append_conventional_generator(conventional_generators,supplemental_generators)

    # print out
    print("Oldest operating year :",int(oldest_year))
    print("Number of active generators :",conventional_generators["nameplate"].size)
    print("Supplemental capacity :",supplemental_capacity)
    print("Capacity removed :",int(total_capacity_removed - supplemental_capacity))
    print("Conventional fleet capacity :",(np.sum(conventional_generators["summer nameplate"])+np.sum(conventional_generators["winter nameplate"]))//2)
    print('Base LOLH :', lolh,flush=True)
    return conventional_generators, hourly_fleet_capacity

def make_supplemental_generators(capacity,efor,temperature_dependent_efor,generator_size):
# Make fleet of small generators generators to provide supplemental capacity    
    supplemental_generators = make_conventional_generator(capacity%generator_size,efor,temperature_dependent_efor)

    for i in range(int(capacity/generator_size)):

        fifty_MW_generator = make_conventional_generator(generator_size,efor,temperature_dependent_efor)
        supplemental_generators = append_conventional_generator(supplemental_generators, fifty_MW_generator)

    return supplemental_generators

def make_conventional_generator(capacity,efor,temperature_dependent_efor):
# Make a dictionary for a single conventional generator
    new_generator = dict()

    new_generator["num units"] = 1
    new_generator["nameplate"] = np.array(capacity)
    new_generator["summer nameplate"] = new_generator["nameplate"]
    new_generator["winter nameplate"] = new_generator["nameplate"]
    new_generator["year"] = np.array(9999)
    new_generator["technology"] = np.array("supplemental")

    if temperature_dependent_efor:
        new_generator["efor"] = np.array([efor,]*8760).reshape(1,8760) #reasonable efor for conventional generator
    else:
        new_generator["efor"] = np.array([efor])

    return new_generator
        
def append_conventional_generator(fleet_conventional_generators,additional_generator):
# Combine two generator dictionaries

    for key in fleet_conventional_generators:
        if key == "efor":
            fleet_conventional_generators[key] = np.concatenate((fleet_conventional_generators[key],additional_generator[key]))
        elif key == "num units":
            fleet_conventional_generators[key] += additional_generator[key]
        else:
            fleet_conventional_generators[key] = np.append(fleet_conventional_generators[key],additional_generator[key])

    return fleet_conventional_generators

def make_RE_generator(generator):
# Convert generator parameters into dictionary of numpy arrays (for function compatibility)
    RE_generator = dict()
    RE_generator["num units"] = 1
    RE_generator["nameplate"] = np.array([generator["nameplate"]])
    RE_generator["summer nameplate"] = np.array([generator["nameplate"]])
    RE_generator["winter nameplate"] = np.array([generator["nameplate"]])
    RE_generator["lat"] = np.array([generator["latitude"]])
    RE_generator["lon"] = np.array([generator["longitude"]])
    RE_generator["efor"] = np.array([generator["efor"]])
    RE_generator["generator type"] = generator["generator type"]

    return RE_generator

def elcc_binary_constraints(binary_trial, lolh, target_lolh, additional_load_max, additional_load_min, added_capacity):
# Constraints for the 'get_elcc' binary search    
    trial_limit_not_met = binary_trial < 20
    convergence_not_met = additional_load_max - additional_load_min > 2 * added_capacity / 100
    reliability_not_met = abs(lolh - target_lolh) > 1e-9
    
    return trial_limit_not_met and convergence_not_met and reliability_not_met

def get_elcc(num_iterations, hourly_fleet_capacity, hourly_added_generator_capacity, fleet_storage, 
                added_storage, hourly_load, added_capacity, fleet_renewable_profile, added_renewable_profile):
    """ Find the ELCC of a generator by adding it to a system and adjusting load until the original reliability is met.

    ...
    Args:
        -------------
        `num_iterations` (int): number of capacity curves to sample for MCS.

        `hourly_fleet_capacity` (ndarray): array of hourly capacity for a number of samples (must correspond to num_iterations)

        `hourly_fleet_capacity` (ndarray): array of hourly capacity of an added generator for a number of samples (must correspond to num_iterations)

        `fleet_storage` (dict): dictionary of storage unit(s). 
        
        `added_storage` (dict): dictionary of storage unit(s) to add. 

        `hourly_load` (ndarray): vector of hourly load. 

        `added_capacity` (float): total capacity added. Used to bound the binary search for the ELCC calculation

        `fleet_renewable_profile` (ndarray): vector of hourly renewable output for the fleet

        `added_renewable_profile` (ndarray): vector of hourly renewable output for the added generator     
    """

    # precision for printing lolh
    precision = int(math.log10(num_iterations))

    # find original reliability
    hourly_storage_capacity = get_hourly_storage_contribution(  num_iterations,hourly_fleet_capacity,hourly_load,
                                                                fleet_storage, fleet_renewable_profile)
    hourly_total_capacity = hourly_fleet_capacity + hourly_storage_capacity
  
    target_lolh, hourly_risk = get_lolh(num_iterations, hourly_total_capacity, hourly_load)

    print("Target LOLH :", round(target_lolh,precision),flush=True)
    print('')

    if DEBUG:
        np.savetxt(OUTPUT_DIRECTORY+'fleet_hourly_risk',hourly_risk)

    # combine fleet storage with generator storage
    all_storage = append_storage(fleet_storage, added_storage)
    combined_renewable_profile = fleet_renewable_profile + added_renewable_profile

    # use binary search to find amount of load needed to match base reliability
    additional_load_max = added_capacity
    additional_load_min = 0
    additional_load = additional_load_max / 2


    if DEBUG:
        # include storage operation
        hourly_storage_capacity = get_hourly_storage_contribution(  num_iterations,
                                                                    hourly_fleet_capacity+hourly_added_generator_capacity,
                                                                    hourly_load,
                                                                    all_storage, combined_renewable_profile)

        # combine contribution from fleet, RE generator, and added storage
        hourly_total_capacity = hourly_fleet_capacity + hourly_storage_capacity + hourly_added_generator_capacity

        lolh, hourly_risk = get_lolh(num_iterations, hourly_total_capacity, hourly_load)

        np.savetxt(OUTPUT_DIRECTORY+'generator_hourly_risk',hourly_risk)

    # include storage operation
    hourly_storage_capacity = get_hourly_storage_contribution(  num_iterations,
                                                                hourly_fleet_capacity+hourly_added_generator_capacity,
                                                                hourly_load+additional_load,
                                                                all_storage, combined_renewable_profile)

    # combine contribution from fleet, RE generator, and added storage
    hourly_total_capacity = hourly_fleet_capacity + hourly_storage_capacity + hourly_added_generator_capacity

    lolh, hourly_risk = get_lolh(num_iterations, hourly_total_capacity, hourly_load + additional_load)
    
    print('Additional Load:',additional_load,'LOLH:',round(lolh,precision))

    if DEBUG:
            print(round(lolh,precision))
            print([i%24 for i in range(len(hourly_risk)) if hourly_risk[i]>0])
            print([i for i in hourly_risk if i>0])

    binary_trial = 0

    while elcc_binary_constraints(binary_trial, lolh, target_lolh, additional_load_max, additional_load_min, added_capacity):
        
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
    
        print('Additional Load:',additional_load,'LOLH:',round(lolh,precision), flush=True)

        # print additional debugging information
        if DEBUG:
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
        error_message = "Threshold not met in 20 binary trials. LOLH: "+str(lolh)
        print(error_message)


    elcc = additional_load

    print('')

    return elcc, hourly_risk

################ PRINT/SAVE/LOAD ######################

def print_parameters(*parameters):
    " Print all parameters."

    print("Parameters:")
    for sub_parameters in parameters:
        for key, value in sub_parameters.items():
            print("\t",key,":",value)

    print('')

def print_fleet(conventional_generators,solar_generators,wind_generators,storage_units):
    " Print information about the fleet found, including storage, conventional, solar, and wind generators."
    # conventional
    print(  "found",conventional_generators["num units"],
            "conventional generators ("+str(int(np.sum(conventional_generators["nameplate"])))+" MW)")

    # renewables
    print(  "found",solar_generators["num units"],"solar generators ("+str(int(np.sum(solar_generators["nameplate"])))+" MW)")

    print(  "found",wind_generators["num units"],"wind generators ("+str(int(np.sum(wind_generators["nameplate"])))+" MW)")

    # storage
    print(  "found", storage_units["num units"],"storage units ("+str(int(np.sum(storage_units["max discharge rate"])))+" MW)")

    print(  "Total Installed Capacity :",   np.sum(conventional_generators["nameplate"])+np.sum(solar_generators["nameplate"])+\
                                            np.sum(wind_generators["nameplate"])+np.sum(storage_units["max discharge rate"]))

    print('')

def save_active_generators(root_directory, conventional, solar, wind, storage, renewable_profile):
    " Save active generators to a csv."

    #conventional
    if conventional['num units'] != 0:
        conventional_generator_array = np.array([   conventional["nameplate"],conventional["summer nameplate"],
                                                    conventional["winter nameplate"],conventional["year"],
                                                    conventional["technology"]])

        conventional_generator_df = pd.DataFrame(   data=conventional_generator_array.T,
                                                    index=np.arange(conventional["nameplate"].size),
                                                    columns=["Nameplate Capacity (MW)", "Summer Capacity (MW)", 
                                                            "Winter Capacity (MW)", "Year", "Technology"])

        conventional_generator_df.to_csv(root_directory+"active_conventional.csv")

    #solar
    if solar['num units'] != 0:

        solar_generator_array = np.array([  solar["nameplate"],solar["summer nameplate"],
                                            solar["winter nameplate"],solar["lat"],
                                            solar["lon"]])

        solar_generator_df = pd.DataFrame(  data=solar_generator_array.T,
                                            index=np.arange(solar["nameplate"].size),
                                            columns=["Nameplate Capacity (MW)","Summer Capacity (MW)",
                                                    "Winter Capacity (MW)","Latitude",
                                                    "Longitude"])
        solar_generator_df.to_csv(root_directory+"active_solar.csv")

    #wind
    if wind['num units'] != 0:

        wind_generator_array = np.array([   wind["nameplate"],wind["summer nameplate"],
                                            wind["winter nameplate"],wind["lat"],
                                            wind["lon"]])

        wind_generator_df = pd.DataFrame(   data=wind_generator_array.T,
                                            index=np.arange(wind["nameplate"].size),
                                            columns=["Nameplate Capacity (MW)","Summer Capacity (MW)",
                                                    "Winter Capacity (MW)","Latitude",
                                                    "Longitude"])
        wind_generator_df.to_csv(root_directory+"active_wind.csv")

    #storage
    if storage['num units'] != 0:
        storage_array = np.array([storage["max charge rate"],storage["max discharge rate"],
                                            storage["max energy"]])
        
        storage_df = pd.DataFrame(  data=storage_array.T,
                                    index=np.arange(storage["max charge rate"].size),
                                    columns=[   "Charge Rate (MW)","Discharge Rate (MW)",
                                                "Nameplate Energy Capacity (MWh)"])
        storage_df.to_csv(root_directory+"active_storage.csv")

    return

def get_saved_system_name(simulation, files, system, create=False):
# Find the "Unique" name for this system. !! poorly-implemented and not very flexible.

    root_directory = files['saved systems folder']

    # level 1 - year
    year = str(simulation['year'])
    root_directory += year + '/'

    if not path.exists(root_directory) and create:
        os.system('mkdir '+root_directory)

    # level 2 - region
    
    region = str(np.sort(simulation['region'])).replace('[','').replace('\'','').replace(',','').replace(']','').replace(' ','_')
    root_directory += region + '/'

    if not path.exists(root_directory) and create:
        os.system('mkdir '+root_directory)

    # level 3 - remaining parameters
    key_words = [   'iterations','target reliability', 'shift load', 'renewable multiplier', 'conventional efor', 'renewable efor',
                    'temperature dependent FOR', 'enable total interchange', 
                    'dispatch strategy', 'storage efficiency', 'supplemental storage',
                    'supplemental storage power capacity', 'supplemental storage energy capacity']
    key_short = [   'its','tgt_rel', 'shift_hrs', 're_mult', 'conv_efor', 're_efor','temp_dep_efor', 'tot_inter',  
                    'disp_strat', 'stor_eff', 'supp_stor','supp_power', 'supp_energy']

    parameters = dict() 

    for group in [simulation, files, system]:
        for key in group:
            if str(key) in key_words:
                parameters[key] = group[key]

    # deal with supplemental storage
    parameters['supplemental storage power capacity'] *= parameters['supplemental storage']
    parameters['supplemental storage energy capacity'] *= parameters['supplemental storage']


    saved_system_directory = root_directory+'system'

    for i in range(len(key_words)):

        saved_system_directory += '__'+key_short[i]+'__'+str(parameters[key_words[i]])

    saved_system_directory += '/'
    return saved_system_directory

def save_hourly_fleet_capacity( hourly_capacity, conventional_generators, solar_generators,
                                wind_generators, storage, renewable_profile, simulation, files, system):
    "Save hourly fleet capacity to a unique system csv." 

    saved_system_directory = get_saved_system_name(simulation,files,system,True)

    if not path.exists(saved_system_directory):
        os.system('mkdir '+saved_system_directory)

    # save components
    np.save(saved_system_directory+'fleet_capacity',hourly_capacity)
    np.save(saved_system_directory+'fleet_renewable_profile', renewable_profile)
    save_active_generators( saved_system_directory,conventional_generators,
                            solar_generators,wind_generators, storage, renewable_profile)

    print("System Saved:\t",str(datetime.datetime.now().time()),flush=True)
    print('')

    return 

def load_hourly_fleet_capacity(simulation,files,system):
    "Load hourly fleet capacity from a unique system csv. If system setting is not \"save\" then return None values."
    saved_system_name = get_saved_system_name(simulation,files,system)

    if not path.exists(saved_system_name) or not system["system setting"] == "save":
        return None, None
    else:
        hourly_capacity = np.load(saved_system_name+'fleet_capacity.npy',allow_pickle=True)
        renewable_profile = np.load(saved_system_name+'fleet_renewable_profile.npy',allow_pickle=True)
        print("System Loaded:\t",str(datetime.datetime.now().time()),flush=True)
        print('')

    return hourly_capacity, renewable_profile

###################### MAIN ############################

def main(simulation,files,system,generator):
    print("Begin Main:\t",str(datetime.datetime.now().time()))
    # initialize global variables
    global DEBUG 
    DEBUG = simulation["debug"]

    # initialize output 
    global OUTPUT_DIRECTORY
    OUTPUT_DIRECTORY = files["output directory"]
    
    # display parameters
    print_parameters(simulation,files,system,generator)

    # get file data
    powGen_lats, powGen_lons, cf = get_powGen(files["solar cf file"],files["wind cf file"])
    hourly_load = get_hourly_load(simulation["year"],simulation["all regions"],simulation["shift load"])
    temperature_data = get_temperature_data(files["temperature file"])
    benchmark_fors = get_benchmark_fors(files["benchmark FORs file"])

    # implements imports/exports for balancing authority
    if system["enable total interchange"]:
        hourly_load += get_total_interchange(simulation["year"],simulation["all regions"],files["total interchange folder"],simulation["shift load"]).astype(np.int64)
    
    # always get storage
    fleet_storage = get_storage_fleet(  files["eia folder"],simulation["all regions"],2019,
                                        system["storage efficiency"],system["storage efor"],system["dispatch strategy"])

    # try loading system
    hourly_fleet_capacity, fleet_renewable_profile = load_hourly_fleet_capacity(simulation, files, system)

    if hourly_fleet_capacity is None:
        # system 
        fleet_conventional_generators = get_conventional_fleet(files["eia folder"], simulation["all regions"],
                                                                2019, system, powGen_lats, powGen_lons,
                                                                temperature_data, benchmark_fors)
        fleet_solar_generators, fleet_wind_generators = get_solar_and_wind_fleet(files["eia folder"],simulation["all regions"],
                                                                                2019, system["renewable efor"],
                                                                                powGen_lats, powGen_lons,system["renewable multiplier"])
        
        print_fleet(fleet_conventional_generators,fleet_solar_generators,fleet_wind_generators,fleet_storage)

        # Supplemental fleet_storage
        fleet_supplemental_storage = make_storage(  system["supplemental storage"],system["supplemental storage energy capacity"],
                                                    system["supplemental storage power capacity"],system["supplemental storage power capacity"],
                                                    system["storage efficiency"],system["storage efor"],system["dispatch strategy"])
        fleet_storage = append_storage(fleet_storage, fleet_supplemental_storage)

        # renewable profile for storage arbitrage
        fleet_renewable_profile = get_RE_profile_for_storage(cf,fleet_solar_generators,fleet_wind_generators)

        # remove generators to find a target reliability level (2.4 loss of load hours per year) and get hourly fleet capacity
        fleet_conventional_generators, hourly_fleet_capacity = remove_generators(   simulation["iterations"],fleet_conventional_generators,
                                                                                    fleet_solar_generators,fleet_wind_generators,fleet_storage,
                                                                                    cf,hourly_load,system["oldest year"],simulation["target reliability"],
                                                                                    system["temperature dependent FOR"],system["conventional efor"], fleet_renewable_profile)

        # option to save system for detailed analysis
        # filename contains simulation parameters
        if system["system setting"] == "save":
            save_hourly_fleet_capacity( hourly_fleet_capacity, fleet_conventional_generators, fleet_solar_generators,
                                        fleet_wind_generators, fleet_storage, fleet_renewable_profile, simulation, files, system)  
            return 0

    # format RE generator 
    RE_generator = make_RE_generator(generator)

    # get cf index
    get_cf_index(RE_generator,powGen_lats,powGen_lons)

    # get hourly capacity matrix
    hourly_RE_generator_capacity = get_hourly_capacity(simulation["iterations"],RE_generator,cf[generator["generator type"]])
    
    # new generator profile for storage arbitrage
    added_renewable_profile = get_RE_profile_for_storage(cf,RE_generator)

    # get added storage
    added_storage = make_storage(   generator["generator storage"],generator["generator storage energy capacity"],
                                    generator["generator storage power capacity"],generator["generator storage power capacity"], 
                                    system["storage efficiency"],system["storage efor"],system["dispatch strategy"])

    # calculate elcc
    added_capacity = generator["nameplate"] + generator["generator storage"]*generator["generator storage power capacity"]
    elcc, hourlyRisk = get_elcc(    simulation["iterations"],hourly_fleet_capacity,hourly_RE_generator_capacity, 
                                    fleet_storage,added_storage, hourly_load, added_capacity, 
                                    fleet_renewable_profile, added_renewable_profile)

    print('**********!!!!!!!!!!!!*********** ELCC :', int(elcc/added_capacity*100),'\n')

    if DEBUG:
        np.savetxt(OUTPUT_DIRECTORY+'hourly_risk.csv',hourlyRisk,delimiter=',')

    print("End Main :\t",str(datetime.datetime.now().time()))
    return elcc