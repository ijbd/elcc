import os

root_directory = '/scratch/mtcraig_root/mtcraig1/shared_data/elccJobs/tests/iterationTest'

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
    
    os.system('sbatch elcc_batch_job.sbat ' + parameter_string)


def main():

    parameters = dict()

    # universal parameters

    parameters['year'] = 2018
    
    # variable parameters
    parameters['iterations'] = 10000

    for count in range(20):
        parameters['count'] = count
        print('Running: count', count)
        run_job(parameters)


if __name__ == "__main__":
    error_handling()
    main()