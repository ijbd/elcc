import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import pandas as pd 

root_directory = '../../archive/2018_PACE_storage_high_res/'
results = pd.read_csv(root_directory+'results.csv')

datum = dict()
datum["storage bool"] = results["supplemental storage"].values
datum["nameplate capacity"] = results["supplemental storage power capacity"].values
datum["duration"] = results["supplemental storage energy capacity"].values / datum["nameplate capacity"]
datum["capacity removed"] = results["Capacity removed"].values
datum["nameplate capacity"] *= datum["storage bool"]
datum["elcc"] = results["ELCC"].values

sort_order = np.argsort(datum["nameplate capacity"])

for key in datum:
    datum[key] = datum[key][sort_order]

# plot elcc vals 

fig, ax = plt.subplots()

for dur in np.unique(datum["duration"]):
    color = (dur/np.max(datum["duration"]),0,0)
    plt_cap = datum["nameplate capacity"][datum["duration"] == dur]
    plt_elcc = datum["elcc"][datum["duration"] == dur]
    ax.plot(plt_cap,plt_elcc,c=color)

ax.set_xlabel('Storage Capacity (MW)')
ax.set_ylabel('ELCC (% of nameplate)')
plt.legend(np.unique(datum["duration"]))
plt.title('ELCC of 100 MW Solar in Salt Lake City\n W/ Variable Storage')

plt.savefig("storage_high_res_elcc_sensitivity")

# plot 
plt.close()

fig, ax = plt.subplots()

for dur in np.unique(datum["duration"]):
    color = (dur/np.max(datum["duration"]),0,0)
    plt_cap = datum["nameplate capacity"][datum["duration"] == dur]
    plt_capacity_removed = datum["capacity removed"][datum["duration"] == dur]
    ax.plot(plt_cap,plt_capacity_removed-500,c=color)

ax.set_xlabel('Storage Capacity (MW)')
ax.set_ylabel('Capacity Removed (MW)')
plt.legend(np.unique(datum["duration"]))
plt.title('Conventional Capacity Offset \n Variable Storage')
plt.savefig("storage_high_res_offset_capacity")