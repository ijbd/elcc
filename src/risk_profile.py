import sys
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from netCDF4 import Dataset 
from elcc_impl import get_hourly_load, get_total_interchange

region = sys.argv[1]
year = 2016

risk_1x = np.loadtxt(region+'_1x_hourly_risk.csv')
risk_1_5x = np.loadtxt(region+'_1.5x_hourly_risk.csv')
risk_2x = np.loadtxt(region+'_2x_hourly_risk.csv')
risk_3x = np.loadtxt(region+'_3x_hourly_risk.csv')
risk_5x = np.loadtxt(region+'_5x_hourly_risk.csv')
risk_diff_1_5x = risk_1_5x - risk_1x
risk_diff_2x = risk_2x - risk_1x
risk_diff_3x = risk_3x - risk_1x
risk_diff_5x = risk_5x - risk_1x

# load

# get correct regions
TEPPC_regions = pd.read_csv('../demand/_regions.csv').fillna('nan')

regions = TEPPC_regions[region.capitalize()].values.flatten().astype(str)
regions = np.unique(regions[regions != 'nan'])

load = get_hourly_load(year,regions) + get_total_interchange(year,regions,'../total_interchange/')

# renewable profile

renewables_1x = np.load(region+'_1x_fleet_renewable_profile.npy',allow_pickle=True)
renewables_1_5x = np.load(region+'_1.5x_fleet_renewable_profile.npy',allow_pickle=True)
renewables_2x = np.load(region+'_2x_fleet_renewable_profile.npy',allow_pickle=True)
renewables_3x = np.load(region+'_3x_fleet_renewable_profile.npy',allow_pickle=True)
renewables_5x = np.load(region+'_5x_fleet_renewable_profile.npy',allow_pickle=True)

# generation

lat = 36
lon = -113.125
lats = np.array(Dataset('../wecc_powGen/2016_solar_generation_cf.nc').variables['lat'][:])
lons = np.array(Dataset('../wecc_powGen/2016_solar_generation_cf.nc').variables['lon'][:])
cf = np.array(Dataset('../wecc_powGen/2016_solar_generation_cf.nc').variables['cf'][:])
cf = cf[np.argwhere(lats==lat)[0],np.argwhere(lons==lon)[0],:].flatten()

# plot
starts = {  'california' : 4960,
            'mountains' : 4980,
            'southwest' : 4930}


start = starts[region]
end = start+90

fig = plt.figure(figsize=(11,7))

ax = fig.add_subplot(111)

ax.plot(load - renewables_1x,lw =3)
ax.plot(load -renewables_1_5x,lw=3)
ax.plot(load -renewables_2x,lw=3)
ax.plot(load -renewables_3x,lw=3)
ax.plot(load -renewables_5x,lw=3)


ax2 = ax.twinx()
ax2.plot(risk_1x,marker='o',ls='--',lw=1,alpha=.5)
ax2.plot(risk_1_5x,marker='o',ls='--',lw=1,alpha=.5)
ax2.plot(risk_2x,marker='o',ls='--',lw=1,alpha=.5)
ax2.plot(risk_3x,marker='o',ls='--',lw=1,alpha=.5)
ax2.plot(risk_5x,marker='o',ls='--',lw=1,alpha=.5)
ax2.plot(cf)


ax.set_xlim([start,end])
ax2.set_ylim([0,1])

ax.legend(['1x Net Load','1.5x Net Load','2x Net Load','3x Net Load','5x Net Load'],loc='upper left')
ax2.legend(['1x Risk','1.5x Risk','2x Risk','3x Risk','5x Risk','Generator Capacity Factor'],loc='upper right')

ax.set_ylabel('Power (MW)')
ax2.set_ylabel('LOLP / Capacity Factor')
ax.set_xlabel('Hour of Year')

ax.set_title('Time Series \n'+region.capitalize()+' '+str(year))

plt.savefig(region+'_risk_profile_b')

# print
'''
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
'''