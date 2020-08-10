import os
import sys
import numpy as np
from elcc_impl import get_powGen

root_directory = sys.argv[1]

LAUNCH_FILE = 'elcc_job_0.txt'

def init():

    global root_directory 

    # In case I forgot a '/'
    if root_directory[-1] != '/':
        root_directory += '/'

    if not os.path.exists(root_directory):
        raise RuntimeError('Invalid root directory\n' + root_directory)

    # Save time and money 
    print("Job Checklist: Have you...")
    user_check("Activated conda environment? (y/n): ",'y')
    user_check("Checked batch resources? (y/n): ",'y')
    user_check("Completed 10 iteration test job? (y/n): ",'y')

    # New launcher file
    new_job()

def user_check(message, test_response):

    status = input(message) == test_response
    if not status:
        sys.exit(1)

def new_job():

    global LAUNCH_FILE

    i = 0
    while os.path.exists('elcc_job_'+str(i)+'.txt'):
        i += 1
    
    LAUNCH_FILE = 'elcc_job_'+str(i)+'.txt'

def add_job(parameters):

    global root_directory
    global LAUNCH_FILE

    # start string with
    parameter_string = ''

    parameters = fix_region_string(parameters)

    for key in parameters:
        parameter_string = parameter_string + ' ' + str(key.replace(' ','_')) + ' ' + str(parameters[key])
    
    with open(LAUNCH_FILE,'a') as f:
        f.write('python -u elcc_driver.py ' + parameter_string + '\n')

def run_job():
# call batch script on current job

    global LAUNCH_FILE

    # only launch non-empty jobs
    if os.path.exists(LAUNCH_FILE):
        # launch job
        os.system('sbatch elcc_batch_job.sbat '+LAUNCH_FILE)

        # start new file for running
        new_job()

def fix_region_string(parameters):
#list to formatted string

    if 'region' in parameters:
        
        regions = parameters['region']

        if isinstance(regions,str):
            return parameters

        #otherwise fix string
        region_str = '\"' + ' '.join(np.sort(regions)) + '\"'

        region_str = region_str[:-1] + '\"'

        parameters['region'] = region_str
    
    return parameters
    

def main():

    parameters = dict()
    parameters['root directory'] = root_directory

    ########### DO NOT WRITE ABOVE THIS LINE (please?) #############

    # universal parameters
    parameters['year'] = 2016
    parameters['region'] = 'Southwest'
    parameters['iterations'] = 5000

    parameters['nameplate'] = 1000
    parameters['generator type'] = 'wind'
    #parameters['generator storage'] = 'True'
    #parameters['generator storage power capacity'] = 500
    #parameters['generator storage energy capacity'] = 500
    
    # variable parameters
    solar_cf_file = "../wecc_powGen/2018_solar_generation_cf.nc" # only used for getting lat/lons
    wind_cf_file = "../wecc_powGen/2018_wind_generation_cf.nc" 

    lats, lons, cf = get_powGen(solar_cf_file, wind_cf_file)

    i = 0 # keep track of job num
    for lat in lats[1::2]: # half resolution
        parameters['latitude'] = lat

        for lon in lons[1::2]: # half resolution
            parameters['longitude'] = lon
            add_job(parameters)
            
            i += 1
            # nine jobs/node
            if i % 9 == 0: run_job()
    
    # run last set if necessary
    run_job()


if __name__ == "__main__":
    init()
    main()