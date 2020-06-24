import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import pandas as pd 

root_directory = '../../archive/2018_PACE_storage_penetration_sensitivity/'
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

for dur in datum["duration"]:
    color=int((dur*331100)%999999)
    plt_capacity = datum["nameplate capacity"][datum["duration"]==dur]
    plt_elcc = datum["elcc"][datum["duration"]==dur]
    ax.plot(plt_capacity/10000*100,plt_elcc,c="#"+str(color))

ax.set_xlabel('Storage Penetration (%)')
ax.set_ylabel('ELCC (% of nameplate)')
plt.legend(['1 Hour Storage', '2 Hour Storage', '3 Hour Storage'])
plt.title('ELCC of 100 MW Solar in Salt Lake City\n Storage Penetration Sensitivity')

plt.savefig("storage_elcc_sensitivity")

# plot 
plt.close()

fig, ax = plt.subplots()

for dur in datum["duration"]:
    color=int((dur*331100)%999999)
    plt_capacity = datum["nameplate capacity"][datum["duration"]==dur]
    plt_elcc = datum["capacity removed"][datum["duration"]==dur]
    ax.plot(plt_capacity,plt_elcc,c="#"+str(color))

ax.set_xlabel('Storage Capacity (MW)')
ax.set_ylabel('Capacity Offset (MW)')
plt.legend(['1 Hour Storage', '2 Hour Storage', '3 Hour Storage'])
plt.title('ELCC of 100 MW Solar in Salt Lake City\n Offset Conventional Capacity Sensitivity')

plt.savefig("storage_offset_capacity")