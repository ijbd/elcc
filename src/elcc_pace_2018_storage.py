import os
import sys
from elcc_impl import get_powGen

root_directory = '/scratch/mtcraig_root/mtcraig1/shared_data/elccJobs/' + sys.argv[1]
RAM = sys.argv[2] #4GB, 8GB (8GB will send email; used for saving systems)

def error_handling():

    global root_directory 

    if root_directory[-1] != '/':
        root_directory += '/'

    if not os.path.exists(root_directory):
        raise RuntimeError('Invalid root directory\n' + root_directory)

    if os.path.exists('elcc_job.txt'):
        os.system('rm elcc_job.txt')

def add_job(parameters):

    global root_directory

    parameter_string = root_directory

    for key in parameters:
        parameter_string = parameter_string + ' ' + str(key) + ' ' + str(parameters[key])
    
    with open('elcc_job.txt','a') as f:
        f.write('python -u elcc_master.py ' + parameter_string + '\n')

def main():

    parameters = dict()

    # universal parameters

    parameters['year'] = 2018
    parameters['region'] = 'PACE'
    parameters['nameplate'] = 1000
    parameters['iterations'] = 10000
    parameters['generator_type'] = 'solar'
    parameters['generator_storage'] = 'True'

    # variable parameters

    solar_cf_file = "../wecc_powGen/2018_solar_generation_cf.nc"
    wind_cf_file = "../wecc_powGen/2018_wind_generation_cf.nc"

    lats, lons, cf = get_powGen(solar_cf_file, wind_cf_file)

    for lat in lats[:1]:
        parameters['latitude'] = lat

        for lon in lons[:1]:
            parameters['longitude'] = lon
            add_job(parameters)

    os.system('sbatch elcc_batch_job_'+RAM+'.sbat')

    print('Jobs submitted')



if __name__ == "__main__":
    error_handling()
    main()