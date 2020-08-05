import sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from netCDF4 import Dataset 

results_filename = sys.argv[1]
img_filename = sys.argv[2]
lines_of_title = sys.argv[3:]

# load results
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

fig, ax = plt.subplots(figsize=(12,10))

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
im = ax.imshow(elcc_map,vmax=vmax,vmin=vmin,cmap='plasma',origin='lower')
cbar = ax.figure.colorbar(im)
cbar.ax.set_ylabel('ELCC (% of Nameplate)',fontsize=15)
cbar.set_ticks(np.linspace(vmin,vmax,5)[1:-1])
cbar.ax.set_yticklabels(np.linspace(vmin,vmax,5)[1:-1],fontsize=12)

tick_max = min_elcc + int((max_elcc - min_elcc)/3)*3 

if (max_elcc - min_elcc) >= 6: 
    #cbar.set_ticks(np.linspace(min_elcc,tick_max,4))
    pass
else:
    #cbar.set_ticks(np.linspace(min_elcc,max_elcc,int(max_elcc-min_elcc)+1))
    pass



ax.set_xlabel('Longitude',fontsize=15)
ax.set_ylabel('Latitude',fontsize=15)

#find number of ticks
num_ticks = np.arange(3,10)
num_lats = np.amin(num_ticks[len(lons)%num_ticks == 0])
num_lons = np.amin(num_ticks[len(lons)%num_ticks == 0])

ax.set_xticks(np.linspace(0,len(lons)-1,num_lons))
ax.set_yticks(np.linspace(0,len(lats)-1,num_lats))
ax.set_yticklabels(np.linspace(lats[0],lats[-1],num_lats),fontsize=12)
ax.set_xticklabels(np.linspace(lons[0],lons[-1],num_lons),fontsize=12)
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


