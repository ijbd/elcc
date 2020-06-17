import numpy as np 
import pandas as pd 
import sys 
import datetime

folder = sys.argv[1]

ts = pd.read_csv(folder+"runtime_profile.csv")

ts_np = ts["Timestamp"].values
hours = []
minutes = []
seconds = []
for timestamp in ts_np:
    hours.append(timestamp.split(":")[0])
    minutes.append(timestamp.split(":")[1])
    seconds.append(timestamp.split(":")[2])
hours = np.array(hours,dtype=int)
minutes = np.array(minutes,dtype=int)
seconds = np.array(seconds,dtype=float)

ts["Hour"] = hours
ts["Minute"] = minutes
ts["Second"] = seconds


ts["next hr"] = np.concatenate((hours[1:],[hours[-1]]))
ts["next min"] = np.concatenate((minutes[1:],[minutes[-1]]))
ts["next sec"] = np.concatenate((seconds[1:],[seconds[-1]]))

ts["Duration Hours"] = (ts["next hr"] - ts["Hour"]) % 60 + (ts["next min"] - ts["Minute"]) // 60
ts["Duration Minutes"] = (ts["next min"] - ts["Minute"]) % 60 + (ts["next sec"] - ts["Second"]) // 60
ts["Duration Seconds"] = (ts["next sec"] - ts["Second"]) % 60

ts.drop(columns=["Timestamp", "next sec", "next min", "next hr"],inplace=True)

unique_funcs = pd.unique(ts["Event"])

ts_unique = pd.DataFrame(unique_funcs, columns=["Function"])

dur_hr = []
dur_min = []
dur_sec = []

for func in ts_unique["Function"]:
    tot_sec = np.sum(ts["Duration Seconds"][ts["Event"] == func].values)
    tot_min = np.sum(ts["Duration Minutes"][ts["Event"] == func].values) + tot_sec // 60
    tot_hrs = np.sum(ts["Duration Hours"][ts["Event"] == func].values) + tot_min // 60

    dur_hr.append(tot_hrs % 24)
    dur_min.append(tot_min % 60)
    dur_sec.append(tot_sec % 60)

ts_unique["Hours"] = dur_hr
ts_unique["Minutes"] = dur_min
ts_unique["Seconds"] = dur_sec

ts_unique.to_csv(folder+"runtime_by_function.csv",index=False)