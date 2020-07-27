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
    parameters['generator_storage'] = True

    # variable parameters
    add_job(parameters)

    os.system('sbatch elcc_batch_job.sbat')

    print('Jobs submitted')



if __name__ == "__main__":
    error_handling()
    main()