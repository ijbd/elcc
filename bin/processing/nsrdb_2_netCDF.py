import pandas as pd 
import numpy as np 
from netCDF4 import Dataset 
import sys

import matplotlib.pyplot as plt 
import matplotlib

year = sys.argv[1]
nsrdb_csv = sys.argv[2]

df = pd.read_csv(nsrdb_csv)
df.drop(df.columns[0], axis=1, inplace=True)

lats = np.array([])
lons = np.array([])
for coord in df.columns:
    lat = float(coord.split(',')[0])
    lon = float(coord.split(',')[1])
    lats = np.append(lats,lat)
    lons = np.append(lons,lon)

lats = np.unique(lats)
lons = np.unique(lons)

mappy = np.zeros((lats.size,lons.size))
for coord in df.columns:
    lat = float(coord.split(',')[0])
    lon = float(coord.split(',')[1])
    mappy[lats==lat,lons==lon] = 1

minlat = np.amin(lats)
minlon = np.amin(lons)
maxlat = np.amax(lats)
maxlon = np.amax(lons)
print(mappy.shape)
plt.contourf(lons,lats,mappy,cmap='binary')
plt.savefig('tmp')
plt.xticks(np.linspace(minlon,maxlon,4))
plt.yticks(np.linspace(minlat,maxlat,4))

if False:
    arr = np.full((lats.size,lons.size,8760),0,dtype=float)

    for coord in df.columns:
        lat = coord.split(',')[0]
        lon = coord.split(',')[1]
        arr[lats==lat,lons==lon] = df[coord]