import numpy as np 
import pandas as pd 
import sys

def get_storage_fleet(include_storage, eia_folder, region, year, round_trip_efficiency, efor, dispatch_strategy):
    
    # return empty storage unit
    if include_storage == False:
        storage = dict()
        storage["num units"] = 0
        return storage
    
    # Open files
    plants = pd.read_excel(eia_folder+"2___Plant_Y2018.xlsx",skiprows=1,usecols=["Plant Code","NERC Region","Latitude",
                                                                                "Longitude","Balancing Authority Code"])
    all_storage_units = pd.read_excel(eia_folder+"3_4_Energy_Storage_Y2018.xlsx",skiprows=1,\
                                                    usecols=["Plant Code","Technology","Nameplate Energy Capacity (MWh)","Status",
                                                            "Operating Year", "Maximum Charge Rate (MW)", "Maximum Discharge Rate (MW)"])
    # Sort by NERC Region and Balancing Authority to filter correct plant codes
    nerc_region_plant_codes = plants["Plant Code"][plants["NERC Region"] == region].values
    balancing_authority_plant_codes = plants["Plant Code"][plants["Balancing Authority Code"] == region].values
    
    desired_plant_codes = np.concatenate((nerc_region_plant_codes, balancing_authority_plant_codes))   

    # Error Handling
    if desired_plant_codes.size == 0:
        error_message = "Invalid region/balancing authority: " + region
        raise RuntimeError(error_message)

    # Get operating generators
    active_storage = all_storage_units[(all_storage_units["Plant Code"].isin(desired_plant_codes)) & (all_storage_units["Status"] == "OP")]

    # Remove empty
    active_storage = active_storage[active_storage["Nameplate Energy Capacity (MWh)"].astype(str) != " "]
    active_storage = active_storage[active_storage["Operating Year"] <= year]
    
    # Fill data structure
    storage = dict()
    storage["dispatch strategy"] = dispatch_strategy
    storage["num units"] = active_storage["Nameplate Energy Capacity (MWh)"].values.size
    storage["max charge rate"] = active_storage["Maximum Charge Rate (MW)"].values
    storage["max discharge rate"] = active_storage["Maximum Discharge Rate (MW)"].values
    storage["max energy"] = active_storage["Nameplate Energy Capacity (MWh)"].values

    # Parametrized (for now)
    storage["roundtrip efficiency"] = np.ones(storage["num units"]) * round_trip_efficiency
    storage["one way efficiency"] = storage["roundtrip efficiency"] ** .5

    # Hourly Tracking (storage starts empty)
    storage["power"] = np.zeros(storage["num units"])
    storage["extractable energy"] = np.ones(storage["num units"])*storage["max energy"]
    storage["energy"] = storage["extractable energy"] / storage["one way efficiency"]
    storage["time to discharge"] = storage["extractable energy"] / storage["max discharge rate"]
    storage["efor"] = efor
    storage["full"] = storage["extractable energy"] == storage["max energy"]

    sys.stdout.flush()

    return storage

def make_storage(include_storage, energy_capacity, charge_rate, discharge_rate,
                 round_trip_efficiency, efor, dispatch_strategy):
    
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
    storage["full"] = storage["extractable energy"] == storage["max energy"]

    return storage

def append_storage(fleet_storage, additional_storage):
      
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

    #for simulation begin empty
    storage["power"] = np.zeros(storage["num units"])
    storage["extractable energy"] = np.ones(storage["num units"])*storage["max energy"] # storage begins full
    storage["energy"] = storage["extractable energy"] / storage["one way efficiency"] 
    storage["time to discharge"] = storage["extractable energy"] / storage["max discharge rate"]

    return

def get_hourly_storage_contribution(num_iterations, hourly_capacity, hourly_load, storage, renewable_profile=None):
    
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

    for day in range(365):
        start = day*24
        end = (day+1)*24
        arbitrage_dispatch(net_load[start:end],storage,hourly_storage_contribution[start:end])
    
    return 

def arbitrage_dispatch(net_load, storage, hourly_storage_contribution):

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

# as implemented by Evans et.al. w/ adjustments by ijbd
def discharge_storage(unmet_load, storage):

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

# based on proposed charging policy by Evans et.al. with adjustments by ijbd
def charge_storage(additional_capacity, storage):

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
        z = y[i-1] + (-P_r - E_min)/(E_max - E_min)*(y[i]-y[i-1])
    
    u = -1*p_d/n*np.maximum(np.minimum(z,z_max)-x,0)

    if storage["efor"] > 0:
        u *= np.random.random_sample(x.shape)>storage["efor"]
        
    #update storage 
    storage["power"] = u
    update_storage(storage, "charge")

    return np.sum(storage["power"])

def update_storage(storage, status):

    if status == "discharge":
        storage["extractable energy"] = storage["extractable energy"] - storage["power"] 
        storage["energy"] = storage["extractable energy"] / storage["one way efficiency"]
    if status == "charge":
        storage["energy"] = storage["energy"] - storage["power"] * storage["one way efficiency"] 
        storage["extractable energy"] = storage["energy"] * storage["one way efficiency"]
    
    # set storage state
    storage["full"] = storage["extractable energy"] == storage["max energy"]

    storage["time to discharge"] = np.divide(storage["extractable energy"],storage["max discharge rate"]) 

    return
