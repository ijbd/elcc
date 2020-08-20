import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from netCDF4 import Dataset 
from elcc_impl import get_hourly_load, get_total_interchange

region = 'California'
year = 2016

risk_1x = np.loadtxt('1x_renewables_hourly_risk.csv')
risk_2x = np.loadtxt('2x_renewables_hourly_risk.csv')
risk_diff = risk_2x - risk_1x

# load

# get correct regions
TEPPC_regions = pd.read_csv('../demand/_regions.csv').fillna('nan')

regions = TEPPC_regions[region].values.flatten().astype(str)
regions = np.unique(regions[regions != 'nan'])

load = get_hourly_load(year,regions) + get_total_interchange(year,regions,'../total_interchange/')

# renewable profile

renewables_1x = np.load('1x_fleet_renewable_profile.npy',allow_pickle=True)
renewables_2x = np.load('2x_fleet_renewable_profile.npy',allow_pickle=True)

# generation

lat = 36
lon = -113.125
lats = np.array(Dataset('../wecc_powGen/2016_solar_generation_cf.nc').variables['lat'][:])
lons = np.array(Dataset('../wecc_powGen/2016_solar_generation_cf.nc').variables['lon'][:])
cf = np.array(Dataset('../wecc_powGen/2016_solar_generation_cf.nc').variables['cf'][:])
cf = cf[np.argwhere(lats==lat)[0],np.argwhere(lons==lon)[0],:].flatten()

# plot

start = 4980
end = 5050

fig = plt.figure(figsize=(11,7))

ax = fig.add_subplot(111)

ax.plot(load,c='#aa0000',lw=5)
ax.plot(renewables_1x)
ax.plot(renewables_2x)
ax2 = ax.twinx()
ax2.plot(risk_1x,marker='x',ls='--',lw=1)
ax2.plot(risk_2x,marker='o',ls=':',lw=1)
ax2.plot(cf)


ax.set_xlim([start,end])
ax2.set_ylim([0,1])

ax.legend(['Load','Original Renewable Profile','2x Renewable Profile'])
ax2.legend(['Original Risk','2x Renewable Risk','Generator Capacity Factor'])


plt.savefig('risk_profile')

# print

#get hours
hours_1x = np.flip(np.argsort(risk_1x))
hours_2x = np.flip(np.argsort(risk_2x))
hours_diff = np.flip(np.argsort(np.abs(risk_diff)))

#get risks
risk_1x = np.flip(np.sort(risk_1x))
risk_2x = np.flip(np.sort(risk_2x))
risk_diff = risk_diff[hours_diff]

print('ORIGINAL SYSTEM')
print('Hours:',hours_1x[:15])
print('Risk:',risk_1x[:15])

print('2X RENEWABLES SYSTEM')
print('Hours:',hours_2x[:15])
print('Risk:',risk_2x[:15])

print('DIFFERENCES')
print('Hours:',hours_diff[:15])
print('Difference:',risk_diff[:15])
#print([risk for risk in risk_diff if risk != 0])




