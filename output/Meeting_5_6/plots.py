import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

w=8
h=6

#1
CISO = pd.read_csv("CISO_raw.csv",header=None)
years = CISO[0].values
elcc = np.around(CISO[1].values,2)

fig,ax = plt.subplots(figsize=(w,h))

x = np.arange(len(years))  # the label locations
width = .5  # the width of the bars
rects = ax.bar(x,elcc,width)

ax.set_ylabel('ELCC (MW)')
ax.set_title('100 MW Solar in Sedona, AZ\nCISO System')
ax.set_xticks(x)
ax.set_xticklabels(years)

autolabel(rects)

plt.savefig("CISO_5_6")


#2
PACE = pd.read_csv("PACE_raw.csv",header=None)

years = PACE[0].values
elcc = np.around(PACE[1].values,2)

fig,ax = plt.subplots(figsize=(w,h))

x = np.arange(len(years))  # the label locations
width = .5  # the width of the bars
rects = ax.bar(x,elcc,width)

ax.set_ylabel('ELCC (MW)')
ax.set_title('100 MW Solar in Salt Lake City, UT\nPACE System')
ax.set_xticks(x)
ax.set_xticklabels(years)

autolabel(rects)

plt.savefig("PACE_5_6")

#3
WECC = pd.read_csv("WECC_raw.csv",header=None)

location = WECC[0].values
elcc = np.around(WECC[1].values,2)
madeini = np.around(WECC[2].values,2)

fig,ax = plt.subplots(figsize=(w,h))

x = np.arange(len(location))  # the label locations
width = .1 # the width of the bars
rects = ax.bar(x-width/2, elcc, width, label="ijbd")
rects2 = ax.bar(x+width/2, madeini, width, label="Madeini & Co.",color='#559999')


ax.set_ylabel('ELCC (MW)')
ax.set_title('Distributed 100 MW Solar\nWECC System 2018')
ax.set_xticks(x)
ax.set_xticklabels(location)
ax.legend()

autolabel(rects)
autolabel(rects2)

plt.savefig("WECC_5_6")


#4
WECC_NP = pd.read_csv("WECC_Nameplate_raw.csv",header=None)

NP = WECC_NP[0].values
elcc = np.around(WECC_NP[1].values/NP*100,2)

fig,ax = plt.subplots(figsize=(w,h))

ax.plot(NP,elcc,Linewidth=2)

ax.set_xlabel('Nameplate Capacity (MW)')
ax.set_xscale('log')
ax.set_ylabel('ELCC (%)')
ax.set_title('Variable Capacity Solar Plant in Seattle, WA\nWECC System 2018')


plt.savefig("WECC_NP_5_6")
