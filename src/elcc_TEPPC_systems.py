import os
import sys

root_directory = '/scratch/mtcraig_root/mtcraig1/shared_data/elccJobs/' + sys.argv[1]

def error_handling():

    global root_directory 

    if root_directory[-1] != '/':
        root_directory += '/'

    if not os.path.exists(root_directory):
        raise RuntimeError('Invalid root directory\n' + root_directory)

    if os.path.exists('elcc_job.txt'):
        os.system('rm elcc_job.txt')

    print("Job Checklist: Have you...")
    status = input("Activated conda environment? (y/n):") == "y"
    status *= input("Checked batch resources? (y/n):") == "y"

    if not status:
        sys.exit(1)

def fix_region_string(parameters):
    if 'region' in parameters:
        
        regions = parameters['region']

        if isinstance(regions,str):
            return parameters

        #otherwise fix string
        region_str = '\"'
        for region in regions:
            region_str += region + ' '

        region_str = region_str[:-1] + '\"'

        parameters['region'] = region_str
    
    return parameters

def add_job(parameters):

    global root_directory

    # start string with
    parameter_string = ''

    parameters = fix_region_string(parameters)

    for key in parameters:
        parameter_string = parameter_string + ' ' + str(key.replace(' ','_')) + ' ' + str(parameters[key])
    
    with open('elcc_job.txt','a') as f:
        f.write('python -u elcc_master.py ' + parameter_string + '\n')

def main():

    parameters = dict()
    parameters['root directory'] = root_directory

    # universal parameters
    parameters['year'] = 2016
    parameters['nameplate'] = 1000
    parameters['iterations'] = 100
    parameters['system_setting'] = 'none'

    # variable parameters
    for region in ['Northwest','Mountains','California','Southwest','Basin']:
        parameters['region'] = region
        for count in range(5):
            parameters['count'] = count
            add_job(parameters)

    os.system('sbatch elcc_batch_job.sbat')

    print('Jobs submitted')



if __name__ == "__main__":
    error_handling()
    main()