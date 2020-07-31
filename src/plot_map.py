import sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from netCDF4 import Dataset 

filename = sys.argv[1]
lines_of_title = sys.argv[2:]

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
vmax = int(max_elcc + (100-max_elcc)/3)

#vmin
vmin = int(min_elcc - (min_elcc)/3)

print(max_elcc,min_elcc)
print(vmax, vmin)
levels = int(vmax-vmin)

# contour plot
cs = ax.contourf(lons,lats,elcc_map,levels=levels,vmax=vmax,vmin=vmin,cmap='plasma')

cbar = fig.colorbar(cs)
cbar.ax.set_ylabel('ELCC (% of Nameplate)')

tick_max = min_elcc + int((max_elcc - min_elcc)/3)*3 

if (max_elcc - min_elcc) >= 6: 
    cbar.set_ticks(np.linspace(min_elcc,tick_max,4))
else:
    cbar.set_ticks(np.linspace(min_elcc,max_elcc,int(max_elcc-min_elcc)+1))


ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_yticks([np.min(lats),40,np.max(lats)])
ax.set_xticks([np.min(lons),-120,-110,np.max(lons)])
ax.grid(True)


title = ''
for line in lines_of_title:
    title += line.replace('\"','') + '\n'

plt.title(title[:-1])

plt.savefig(filename[:-4],bbox_inches='tight',dpi=100)

