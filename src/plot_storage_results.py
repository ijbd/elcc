import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import pandas as pd 

root_directory = '../../archive/2018_PACE_3000MW_Storage/'
results = pd.read_csv(root_directory+'results.csv')

datum = dict()
datum["storage bool"] = results["supplemental storage"].values
datum["nameplate capacity"] = results["supplemental storage power capacity"].values
datum["duration"] = results["supplemental storage energy capacity"].values / datum["nameplate capacity"]
datum["capacity removed"] = results["Capacity removed"].values
datum["nameplate capacity"] *= datum["storage bool"]
datum["elcc"] = results["ELCC"].values

sort_order = np.argsort(datum["duration"])

for key in datum:
    datum[key] = datum[key][sort_order]

# plot elcc vals 

fig, ax = plt.subplots()

ax.plot(datum["duration"],datum["elcc"],c='r')

ax.set_xlabel('Storage Duration (Hours)')
ax.set_ylabel('ELCC (% of nameplate)')
plt.title('ELCC of 100 MW Solar in Salt Lake City\n 3000 MW Storage')

plt.savefig("storage_duration_elcc_sensitivity")

# plot 
plt.close()

fig, ax = plt.subplots()

ax.plot(datum["duration"],datum["capacity removed"],c='r')

ax.set_xlabel('Storage Duration (Hours)')
ax.set_ylabel('Capacity Offset (MW)')
plt.title('Capacity offset of 3000 MW storage')

plt.savefig("storage_duration_offset_capacity")