# -*- coding: utf-8 -*-
"""
Created on Sun May 24 12:47:45 2020

@author: julian
"""
from netCDF4 import Dataset
from datetime import date, timedelta
import datetime
import numpy as np
import datetime
import time as stopWatch
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import sys

#inputs to method should be year then region either wecc or .....


#shapeFinderFilePath = 'MERRA2_400.tavg1_2d_slv_Nx.20160101.nc4.nc4'
#temporary fix for finding correct lat long lengths
shapeFinderFilePath = '../../scratch/mtcraig_root/mtcraig1/shared_data/merraData/resource/%s/raw/MERRA2_400.tavg1_2d_slv_Nx.20170101.nc4.nc4' % (sys.argv[2])

shapeData = Dataset(shapeFinderFilePath)

print("Shape of data: %s" % (str(shapeData.variables["T2M"].shape)))
print("Lat: %s " % (shapeData.variables["T2M"].shape[1]))
print("Lon: %s " % (shapeData.variables["T2M"].shape[2]))

#once assigned don't change this variable important for writing to the same shape for bounds
latLength = shapeData.variables["T2M"].shape[1]
lonLength = shapeData.variables["T2M"].shape[2]

def rewriteData(data1, variableTag):
    data = np.zeros((24,latLength,lonLength))
    timeValueIndex = -1
    latIndex = 0
    longIndex = 0
    for time in data1.variables[variableTag]:
        timeValueIndex += 1
        latIndex = 0
        for lat in time:
            longIndex = 0
            for long in lat:
                data[timeValueIndex][latIndex][longIndex] = long    
                longIndex += 1
            latIndex += 1
    return data

def main(year):
    '''
    args: 
    
    year, for which year you are extracting the temperature data
    
    can only do 1 year increments for now
    '''
    
    
    #shapeFinderFilePath = "../../scratch/mtcraig_root/mtcraig1/shared_data/merraData/resource/wecc/raw/MERRA2_400.tavg1_2d_slv_Nx."
    ''''
    shapeFinderFilePath = "MERRA2_400.tavg1_2d_slv_Nx.20160101.nc4.nc4"
    #tester = "MERRA2_400.tavg1_2d_slv_Nx.20160108.nc4.nc4"
    shapeData = Dataset(shapeFinderFilePath)
    test = np.array(shapeData.variables["T2M"][:][:][:]).T
    hourData2 = test[15,15]
    plt.plot(hourData2[0:24])
    plt.show()
    ax = sns.heatmap(shapeData.variables["T2M"][0])
    plt.show()
    
        
    tester2 = "temperatureDataset2016.nc"    
    currentInUse = np.array(Dataset(tester2)["T2M"][:][:][:]).T
    hourData = currentInUse[15,15]
    plt.plot(hourData[0:24])
    plt.show()
    ax = sns.heatmap((Dataset(tester2)["T2M"][0]))
    plt.show()
    '''
    
    baseWord = "../../scratch/mtcraig_root/mtcraig1/shared_data/merraData/resource/wecc/raw/MERRA2_400.tavg1_2d_slv_Nx."
    fileName = "temperatureDataset%s.nc" % (year)#change for new year or month!!!
    #start date for dataset, may need to change
    start_date = datetime.date(year, 1, 1)

    checkpoint = datetime.date(year, 6, 1)
    #end date for dataset,may need to change
    end_date = datetime.date(year, 12, 31)
    #end_date = datetime.date(year, 1, 1)
    print("Started reading!")
    delta = timedelta(days=1)
    current_date = start_date


    #calculating extracting data length time
    start_timeRewrite = stopWatch.time()



    while current_date <= end_date:
        if current_date == checkpoint:
            print("made it half way")
            print(datetime.datetime.now())
        #checking for leap day then incremanting to the next day (skipping it) if it is a leap day
        if(current_date.year % 4 == 0 and current_date.month == 2 and current_date.day == 29):
            current_date += delta
        date = current_date.strftime("%Y%m%d")
        filename = baseWord + date + ".nc4.nc4"
        netCDF = Dataset(filename)
        #netCDF = Dataset('MERRA2_400.tavg1_2d_slv_Nx.20160101.nc4.nc4')
        if current_date == start_date:
            tempDataset = rewriteData(netCDF,"T2M")
        else:
            tempDataset = np.concatenate((tempDataset,rewriteData(netCDF,"T2M")),axis=0)
        current_date += delta

    print(tempDataset.shape)
    print("made it to creating new netcdf!")
    print(np.max(tempDataset))

    newNetCDF = Dataset(fileName, 'w')
    newNetCDF.createDimension('hour',8760)
    #newNetCDF.createDimension('hour',24)

    newNetCDF.createDimension('lat', latLength)
    newNetCDF.createDimension('long',lonLength)
    varT2M = newNetCDF.createVariable("T2M",'double', ('hour','lat','long'))

    varT2M[:,:,:] = tempDataset[:][:][:]
    print("Done!!")
    #Runtime for program
    rewriteTimeLength = (stopWatch.time() - start_timeRewrite)
    print("It took " + str(round(rewriteTimeLength,2))  + " seconds to rewrite older datasets!")#time to rewrite datsets and write to new file

print("Running year: %s" % (sys.argv[1]))
main(int(sys.argv[1]))
