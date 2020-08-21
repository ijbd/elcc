import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from elcc_impl import get_powGen, get_solar_and_wind_fleet

def get_conventional_fleet_impl(plants,active_generators,year):
# Filter generators for operation status, technology, year of simulation

    # filtering
    active_generators = active_generators[(active_generators["Operating Year"] <= year)]
    active_generators = active_generators[(active_generators["Status"] == "OP")]
    active_generators = active_generators[(~active_generators["Technology"].isin(["Solar Photovoltaic", "Onshore Wind Turbine", "Offshore Wind Turbine", "Batteries"]))]
    
    # Fill empty summer/winter capacities
    active_generators["Summer Capacity (MW)"].where(active_generators["Summer Capacity (MW)"].astype(str) != " ",
                                                    active_generators["Nameplate Capacity (MW)"], inplace=True)
    active_generators["Winter Capacity (MW)"].where(active_generators["Winter Capacity (MW)"].astype(str) != " ", 
                                                    active_generators["Nameplate Capacity (MW)"], inplace=True)
    
    # Convert Dataframe to Dictionary of numpy arrays
    conventional_generators = dict()
    conventional_generators["num units"] = active_generators["Nameplate Capacity (MW)"].values.size
    conventional_generators["nameplate"] = active_generators["Nameplate Capacity (MW)"].values
    conventional_generators["summer nameplate"] = active_generators["Summer Capacity (MW)"].values
    conventional_generators["winter nameplate"] = active_generators["Winter Capacity (MW)"].values
    conventional_generators["year"] = active_generators["Operating Year"].values
    conventional_generators["technology"] = active_generators["Technology"].values

    if conventional_generators["nameplate"].size == 0:
        error_message = "No existing conventional found."
        raise RuntimeError(error_message)
    
    return conventional_generators

def get_conventional_fleet(eia_folder, regions, year):
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
    # Sort by NERC Region and Balancing Authority to filter correct plant codes
    nerc_region_plant_codes = plants["Plant Code"][plants["NERC Region"].isin(regions)].values
    balancing_authority_plant_codes = plants["Plant Code"][plants["Balancing Authority Code"].isin(regions)].values
    
    desired_plant_codes = np.concatenate((nerc_region_plant_codes, balancing_authority_plant_codes))

    # Error Handling
    if desired_plant_codes.size == 0:
        error_message = "Invalid region(s): " + str(regions)
        raise RuntimeError(error_message)

    # Get operating generators
    active_generators = all_conventional_generators[(all_conventional_generators["Plant Code"].isin(desired_plant_codes))]

    return get_conventional_fleet_impl(plants,active_generators,year)

def main():
    year = 2018
    regions = ['Basin','California','Mountains','Northwest','Southwest']
    eia_folder = '../eia8602018/'

    TEPPC_regions = pd.read_csv('../demand/_regions.csv').fillna('nan')

    fleet = pd.DataFrame(columns=regions,index=['Conventional (MW)','Solar (MW)','Wind (MW)','Total (MW)','Conventional (%)','Solar (%)','Wind (%)'])
    
    # powGen
    powGen_lats, powGen_lons, cf = get_powGen('../wecc_powGen/2016_solar_generation_cf.nc','../wecc_powGen/2016_wind_generation_cf.nc')

    # map setup
    fig, ((solar_ax, wind_ax), (solar_idx_ax, wind_idx_ax), (solar_cf_ax, wind_cf_ax))= plt.subplots(nrows=3, ncols=2,figsize=(20,20))

    solar_cf_ax.imshow(np.average(cf['solar'],axis=2),aspect='auto',alpha=.6,cmap='plasma',origin='lower')
    wind_cf_ax.imshow(np.average(cf['wind'],axis=2),aspect='auto',alpha=.6,cmap='plasma',origin='lower')

    for region in regions:

        all_regions = np.array([TEPPC_regions[region].values.flatten().astype(str)])
        all_regions = np.unique(all_regions[all_regions != 'nan'])

        conv = get_conventional_fleet(eia_folder,all_regions,year)
        solar, wind = get_solar_and_wind_fleet(eia_folder, all_regions, year, 1, powGen_lats, powGen_lons, 1)
        c = np.sum(conv['nameplate'])
        s = np.sum(solar['nameplate'])
        w = np.sum(wind['nameplate'])
        t = c+s+w
        fleet.at['Conventional (MW)',region] = round(c)
        fleet.at['Solar (MW)',region] = round(s)
        fleet.at['Wind (MW)',region] = round(w)
        fleet.at['Total (MW)',region] = round(t)
        fleet.at['Conventional (%)',region] = round(c/t*100)
        fleet.at['Solar (%)',region] = round(s/t*100)
        fleet.at['Wind (%)',region] = round(w/t*100)

        solar_ax.scatter(solar['lon'],solar['lat'])
        wind_ax.scatter(wind['lon'],wind['lat'])

        solar_idx_ax.scatter(powGen_lons[solar['lon idx']],powGen_lats[solar['lat idx']])
        wind_idx_ax.scatter(powGen_lons[wind['lon idx']],powGen_lats[wind['lat idx']])

        solar_cf_ax.scatter(solar['lon idx'],solar['lat idx'])
        wind_cf_ax.scatter(wind['lon idx'],wind['lat idx'])

    for ax in [solar_ax, wind_ax, solar_idx_ax, wind_idx_ax, solar_cf_ax, wind_cf_ax]:
        ax.legend(regions)
        if ax != solar_cf_ax and ax != wind_cf_ax:
            ax.set_xlim([np.min(powGen_lons),np.max(powGen_lons)])
            ax.set_ylim([np.min(powGen_lats),np.max(powGen_lats)])
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
        if ax == solar_cf_ax or ax == wind_cf_ax:
            ax.set_xlabel('Nearest Capacity Factor Longitude Index')
            ax.set_ylabel('Nearest Capacity Factor Latitude Index')

    solar_ax.set_title('Solar Plant Locations')
    wind_ax.set_title('Wind Plant Locations')
    solar_idx_ax.set_title('Solar Plant Closest CF Locations')
    wind_idx_ax.set_title('Wind Plant Closest CF Locations')
    solar_cf_ax.set_title('Solar Plant Closest CF Location over Average Solar CF')
    wind_cf_ax.set_title('Wind Plant Closest CF Location over Average Wind CF')

    # wrap up

    plt.savefig('regional_composition_map')

    fleet.to_csv('regional_composition.csv')


    


    print('All Finished!')


if __name__ == "__main__":
    main()