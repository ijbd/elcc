import os
import sys

root_directory = '/scratch/mtcraig_root/mtcraig1/shared_data/elccJobs/' + sys.argv[1]
RAM = sys.argv[2] #4GB, 8GB (8GB will send email; used for saving systems)

def error_handling():

    global root_directory 

    if root_directory[-1] != '/':
        root_directory += '/'
    if not os.path.exists(root_directory):
        raise RuntimeError('Invalid root directory\n' + root_directory)

def run_job(parameters):

    global root_directory

    parameter_string = str(root_directory)

    for key in parameters:
        parameter_string = parameter_string + ' ' + str(key) + ' ' + str(parameters[key])
    
    os.system('sbatch elcc_batch_job_'+RAM+'.sbat ' + parameter_string)

def main():

    parameters = dict()

    # universal parameters
    parameters['nameplate'] = 1000
    parameters['iterations'] = 10000
    parameters['generator_type'] = 'solar'

    # variable parameters

    
    for year in [2016, 2017, 2018]:
        parameters['year'] = year
        for region in ['CISO','IPCO','WACM','WECC']:
            parameters['region'] = region

            print('Running:',region,year)
            run_job(parameters)
    

if __name__ == "__main__":
    error_handling()
    main()