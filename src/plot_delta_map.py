import sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from netCDF4 import Dataset 
import matplotlib.colors as colors 

# difference between filename 1 and 2
results_filename_1 = sys.argv[1]
results_filename_2 = sys.argv[2]
img_filename = sys.argv[3]
lines_of_title = sys.argv[4:]

# load results

def get_elcc_map(results_filename, **kwargs):

    results = pd.read_csv(results_filename,index_col=0)

    latitude = results['latitude'].values.astype(float)
    longitude = results['longitude'].values.astype(float)
    elcc = results['ELCC'].values.astype(int)

    # make map 

    lats = np.unique(latitude)
    lons = np.unique(longitude)

    elcc_map = np.zeros((len(lats),len(lons)))

    # fill map

    for i in range(len(elcc)):
        
        elcc_map[np.argwhere(lats == latitude[i])[0,0], np.argwhere(lons == longitude[i])[0,0]] = elcc[i]

    if "return_lat_lons" in kwargs:
        if kwargs["return_lat_lons"]:
            return elcc_map, lats, lons

    return elcc_map

elcc_map_1,lats,lons = get_elcc_map(results_filename_1,return_lat_lons=True)
elcc_map_2 = get_elcc_map(results_filename_2)
elcc_map = elcc_map_1 - elcc_map_2

fig, ax = plt.subplots(figsize=(12,10))

# find ranges
max_elcc = np.amax(elcc_map)
min_elcc = np.amin(elcc_map)

# contour plot
biggest_diff = np.maximum(abs(max_elcc),abs(min_elcc))

divnorm = colors.TwoSlopeNorm(vmin=-1*abs(biggest_diff), vcenter=0, vmax=abs(biggest_diff))
im = ax.imshow(elcc_map,cmap='coolwarm',origin='lower',norm=divnorm)
cbar = ax.figure.colorbar(im)
cbar.ax.set_ylabel('$\Delta$ ELCC',fontsize=15)
cbar.set_ticks(np.linspace(-biggest_diff,biggest_diff,3))


ax.set_xlabel('Longitude',fontsize=15)
ax.set_ylabel('Latitude',fontsize=15)

ax.set_xticks(np.linspace(0,len(lons)-1,3))
ax.set_yticks(np.linspace(0,len(lats)-1,3))
ax.set_yticklabels(np.linspace(lats[0],lats[-1],3),fontsize=12)
ax.set_xticklabels(np.linspace(lons[0],lons[-1],3),fontsize=12)
#ax.grid(True)

for i in range(len(lats)):
    for j in range(len(lons)):
        text = ax.text(j, i, str(int(elcc_map[i, j])),
                       ha="center", va="center", color="w")


title = ''
for line in lines_of_title:
    title += line.replace('\"','') + '\n'

plt.title(title[:-1],fontsize=18)

plt.savefig(img_filename,bbox_inches='tight',dpi=100)


