import os
import sys
import numpy as np

root_directory = '/scratch/mtcraig_root/mtcraig1/shared_data/elccJobs/' + sys.argv[1]

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
    user_check("Activated conda environment? (y/n):")
    user_check("Checked batch resources? (y/n):")

    # New launcher file
    new_job()

def user_check(message):

    status = input(message) == "y"
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
        f.write('python -u elcc_master.py ' + parameter_string + '\n')

def run_job():

    global LAUNCH_FILE

    # launch job
    os.system('sbatch elcc_batch_job.sbat '+LAUNCH_FILE)

    # start new file for running
    new_job()

def fix_region_string(parameters):

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
    parameters['nameplate'] = 1000
    parameters['system_setting'] = 'none'

    # variable parameters
    for iterations in [10]:

        parameters['iterations'] = iterations
        for region in ['Northwest']:#,'Mountains','California','Southwest','Basin']:
            
            parameters['region'] = region
            for count in range(1):#5):
                parameters['count'] = count
                add_job(parameters)

            #run jobs by region
            run_job()



if __name__ == "__main__":
    init()
    main()