import pandas as pd 
import numpy as np
import sys

from elcc_impl import add_partial_ownership_generators

year = int(sys.argv[1])
region = sys.argv[2].split()
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

plants = all_plants[(all_plants["Balancing Authority Code"].isin(region)) | (all_plants["NERC Region"].isin(region))]

# conventional
generators = all_generators[all_generators['Plant Code'].isin(plants["Plant Code"])]
generators = add_partial_ownership_generators(eia_folder,region,year,generators,all_generators)

#filtering
renewables = ["Solar Photovoltaic", "Onshore Wind Turbine", "Offshore Wind Turbine", "Batteries"]

generators = generators[~generators['Technology'].isin(renewables)]
generators = generators[generators['Status'] == 'OP']
generators = generators[generators['Operating Year'] <= year]
generators["Summer Capacity (MW)"].where(   generators["Summer Capacity (MW)"].astype(str) != " ",
                                            generators["Nameplate Capacity (MW)"], inplace=True)
generators["Winter Capacity (MW)"].where(   generators["Winter Capacity (MW)"].astype(str) != " ", 
                                            generators["Nameplate Capacity (MW)"], inplace=True)                              



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

generators.to_csv('_'.join(region)+'_'+str(year)+'_generators.csv')



