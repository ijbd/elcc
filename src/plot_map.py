import sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from netCDF4 import Dataset 

filename = sys.argv[1]
plot_title = str(sys.argv[2].replace('\"',''))

# load results
results = pd.read_csv(filename,index_col=0)

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

fig, ax = plt.subplots()

# find ranges
max_elcc = np.amax(elcc_map)
min_elcc = np.amin(elcc_map)

#vmax 
vmax = max_elcc + 15

if max_elcc > 50:
    vmax = max_elcc + 5

if max_elcc > 75:
    vmax = max_elcc

#vmin
vmin = min_elcc -15

if min_elcc < 50:
    vmin = min_elcc -5

if min_elcc < 25:
    vmin = min_elcc

levels = int(vmax-vmin)

# contour plot
cs = ax.contourf(lons,lats,elcc_map,levels=levels,vmax=vmax,vmin=vmin,cmap='plasma')

cbar = fig.colorbar(cs)
cbar.ax.set_ylabel('ELCC (% of Nameplate)')
cbar.set_ticks(np.linspace(min_elcc,max_elcc,int(levels/4)))


ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_yticks([np.min(lats),40,np.max(lats)])
ax.set_xticks([np.min(lons),-120,-110,np.max(lons)])
ax.grid(True)

plt.title(plot_title)

plt.savefig(filename[:-4],bbox_inches='tight',dpi=100)


