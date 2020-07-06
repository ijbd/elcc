import os

root_directory = '/scratch/mtcraig_root/mtcraig1/shared_data/elccJobs/tests/'

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
    
    os.system('echo sbatch elcc_batch_job.sbat ' + parameter_string)


def main():

    parameters = dict()

    #### SETUP JOB

    parameters['year'] = 2018
    parameters['region'] = 'PACE'

    run_job(parameters)



if __name__ == "__main__":
    error_handling()
    main()