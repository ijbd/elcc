import os
import sys

root_directory = '/scratch/mtcraig_root/mtcraig1/shared_data/elccJobs/' + sys.argv[1]
email = False

if len(sys.argv) == 3:
    if sys.argv[2] == "email":
        email = True

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
    
    if email:
        os.system('sbatch elcc_batch_job_email.sbat ' + parameter_string)
    else:
        os.system('sbatch elcc_batch_job.sbat ' + parameter_string)

def main():

    parameters = dict()

    # universal parameters

    parameters['year'] = 2018
    parameters['region'] = 'PACE'
    parameters['nameplate'] = 100
    parameters['iterations'] = 10000
    parameters['generator_type'] = 'solar'

    # variable parameters
    for count in range(5):
        parameters['count'] = count
        run_job(parameters)

if __name__ == "__main__":
    error_handling()
    main()