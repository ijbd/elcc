import os

root_directory = '/scratch/mtcraig_root/mtcraig1/shared_data/elccJobs/tests/'

def run_job(parameters):
    parameter_string = str(root_directory)

    for key in parameters:
        parameter_string = parameter_string + ' ' + str(key) + ' ' + str(parameters[key])
    
    os.system('sbat elcc_batch_job.sbat' + parameter_string)


def main():

    if not os.path.exists(root_directory):
        raise RuntimeError('Invalid root_directory')

    parameters = dict()

    #### SETUP JOB

    parameters['year'] = 2018
    parameters['region'] = 'PACE'

    run_job(parameters)



if __name__ == "__main__":
    main()