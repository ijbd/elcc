import pandas as pd 
import numpy as np
import sys

year = int(sys.argv[1])
region = sys.argv[2]
eia_folder = '../eia860'+str(year)+'/'

all_plants = pd.read_excel(eia_folder+"2___Plant_Y"+str(year)+".xlsx",skiprows=1,usecols=["Plant Code","NERC Region","Balancing Authority Code"])
all_generators = pd.read_excel(eia_folder+"3_1_Generator_Y"+str(year)+".xlsx",skiprows=1,
                                                    usecols=["Plant Code","Plant Name","Generator ID","Nameplate Capacity (MW)","Status","Technology",
                                                            "Operating Year", "Summer Capacity (MW)", "Winter Capacity (MW)"])

all_solar = pd.read_excel(eia_folder+"3_3_Solar_Y"+str(year)+".xlsx",skiprows=1,
                                                    usecols=["Plant Code","Plant Name","Generator ID","Nameplate Capacity (MW)","Status","Technology",
                                                            "Operating Year", "Summer Capacity (MW)", "Winter Capacity (MW)"])

all_wind = pd.read_excel(eia_folder+"3_2_Wind_Y"+str(year)+".xlsx",skiprows=1,
                                                    usecols=["Plant Code","Plant Name","Generator ID","Nameplate Capacity (MW)","Status","Technology",
                                                            "Operating Year", "Summer Capacity (MW)", "Winter Capacity (MW)"])  

# get plant_codes

plants = all_plants[(all_plants["Balancing Authority Code"] == region) | (all_plants["NERC Region"] == region)]

# conventional
renewables = ["Solar Photovoltaic", "Onshore Wind Turbine", "Offshore Wind Turbine", "Batteries"]

generators = all_generators[~all_generators['Technology'].isin(renewables)]
generators = generators[generators['Plant Code'].isin(plants["Plant Code"])]
generators = generators[generators['Status'] == 'OP']
generators = generators[generators['Operating Year'] <= year]
generators["Summer Capacity (MW)"].where(   generators["Summer Capacity (MW)"].astype(str) != " ",
                                            generators["Nameplate Capacity (MW)"], inplace=True)
generators["Winter Capacity (MW)"].where(   generators["Winter Capacity (MW)"].astype(str) != " ", 
                                            generators["Nameplate Capacity (MW)"], inplace=True)

# joint ownership
def add_partial_ownership_generators(eia_folder,region,year,generators,all_generators,renewable=False):
        if region in ["AZPS","PSCO"]:
                utilities = {"AZPS" : "Arizona Public Service Co","PSCO" : "Public Service Co of Colorado"}
                utility_name = utilities[region]

                owners = pd.read_excel(eia_folder+"4___Owner_Y"+str(year)+".xlsx",skiprows=1,usecols=[  "Plant Code","Generator ID",
                                                                                                        "Status","Owner Name","Percent Owned"])

                # filtering
                owners = owners[owners["Status"] == "OP"]
                owners = owners[owners["Owner Name"] == utility_name]
                generators= generators[~generators["Plant Code"].isin(owners["Plant Code"])]

                if not renewable:
                        renewables = ["Solar Photovoltaic", "Onshore Wind Turbine", "Offshore Wind Turbine", "Batteries"]
                        all_generators = all_generators[~all_generators['Technology'].isin(renewables)]

                if not owners.empty:
                        for ind, row in owners.iterrows():
                                generator = all_generators[     (all_generators["Plant Code"] == row["Plant Code"]) &\
                                                                (all_generators["Generator ID"] == row["Generator ID"])]
                                partial_generator = generator.copy()
                                idx = partial_generator.index[0]                       
                                partial_generator.at[idx,"Nameplate Capacity (MW)"] *= row["Percent Owned"]
                                partial_generator.at[idx,"Summer Capacity (MW)"] *= row["Percent Owned"]
                                partial_generator.at[idx,"Winter Capacity (MW)"] *= row["Percent Owned"]
                                generators = generators.append(partial_generator)
                
                return generators
                                

generators = add_partial_ownership_generators(eia_folder,region,year,generators,all_generators,True)

# solar
solar = all_solar[all_solar['Plant Code'].isin(plants["Plant Code"])]
solar = solar[solar['Status'] == 'OP']
solar = solar[solar['Operating Year'] <= year]
solar["Summer Capacity (MW)"].where(    solar["Summer Capacity (MW)"].astype(str) != " ",
                                        solar["Nameplate Capacity (MW)"], inplace=True)
solar["Winter Capacity (MW)"].where(    solar["Winter Capacity (MW)"].astype(str) != " ", 
                                        solar["Nameplate Capacity (MW)"], inplace=True)

# wind 
wind = all_wind[all_wind['Plant Code'].isin(plants["Plant Code"])]
wind = wind[wind['Status'] == 'OP']
wind = wind[wind['Operating Year'] <= year]
wind["Summer Capacity (MW)"].where(     wind["Summer Capacity (MW)"].astype(str) != " ",
                                        wind["Nameplate Capacity (MW)"], inplace=True)
wind["Winter Capacity (MW)"].where(     wind["Winter Capacity (MW)"].astype(str) != " ", 
                                        wind["Nameplate Capacity (MW)"], inplace=True)

# print
solar_nameplate = np.sum(solar["Nameplate Capacity (MW)"])
solar_summer = np.sum(solar["Summer Capacity (MW)"])
solar_winter = np.sum(solar["Winter Capacity (MW)"])

wind_nameplate = np.sum(wind["Nameplate Capacity (MW)"])
wind_summer = np.sum(wind["Summer Capacity (MW)"])
wind_winter = np.sum(wind["Winter Capacity (MW)"])

conv_nameplate = np.sum(generators["Nameplate Capacity (MW)"])
conv_summer = np.sum(generators["Summer Capacity (MW)"])
conv_winter = np.sum(generators["Winter Capacity (MW)"])

print('Region:','\t','\t',region)
print('Year:','\t','\t','\t',year)
print('Conventional:','\t','\t',int(conv_nameplate))
print('Solar:','\t','\t','\t',int(solar_nameplate))
print('Wind:','\t','\t','\t',int(wind_nameplate))
print('Total:','\t','\t','\t',int(conv_nameplate+solar_nameplate+wind_nameplate))
print('Summer:','\t','\t',int(conv_summer+solar_summer+wind_summer))
print('Winter:','\t','\t',int(conv_winter+solar_winter+wind_winter))
print('Pct. RE:','\t','\t',round((solar_nameplate+wind_nameplate)/(solar_nameplate+wind_nameplate+conv_nameplate)*100),"%")

# save
generators = generators.append(solar)
generators = generators.append(wind)

generators.to_csv(region+'_'+str(year)+'_generators.csv')





