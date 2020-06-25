import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os

root_directory = "../../archive/2018_PACE_storage_penetration_sensitivity/"

def plot_risk(root, filename):
    
    # get data
    hourly_risk = pd.read_csv(root+'/'+filename).values

    # plot 
    fig, ax = plt.subplots()

    ax.plot(hourly_risk)
    ax.set_xlabel("Hour of Year")
    ax.set_ylabel("LOLP")
    ax.set_yticks(np.linspace(0,.6,4))
    plt.title("Hourly Risk")

    # save
    plt.savefig(root+'/'+"hourlyRiskPlot")
    plt.close()

def plot_storage(root, filename):
    
    # get data
    storage = pd.read_csv(root+'/'+filename).values[:,1]

    # plot 
    fig, ax = plt.subplots()

    ax.plot(storage)
    ax.set_xlabel("Hour of Year")
    ax.set_ylabel("Power (MW)")
    plt.title("Storage Contribution")
    
    #save
    plt.savefig(root+'/'+"storage_contribution_plot")
    plt.close()


def main():
    for root, dirs, files in os.walk(root_directory):
            for filename in files:
                if filename.endswith("hourlyRisk.csv"):
                    plot_risk(root, filename)
                if filename.endswith("storage_dbg.csv"):
                    plot_storage(root, filename)

main()
                