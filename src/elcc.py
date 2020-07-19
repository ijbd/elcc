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

def add_job(parameters):

    global root_directory
    global num_jobs

    num_jobs += 1
    parameter_string = str(job)+' '+root_directory

    for key in parameters:
        parameter_string = parameter_string + ' ' + str(key) + ' ' + str(parameters[key])
    
    with open('elcc_job.txt','a') as f:
        #f.write('sbatch elcc_batch_job.sbat ' + parameter_string + '\n')
        f.write('bash elcc_batch_job.sh' + parameter_string + '\n')

def run_jobs(jobs_per_batch):
    i = 0
    jobs_completed = 0

    if not os.path.exists('elcc_job.txt'):
        sys.exit(1)

    with open('elcc_job.txt','r') as f:
        for line in f:
            if jobs_completed >= jobs_per_batch:
                sys.exit(0)
            if i == job:
                os.system(line)
                job += 1
                jobs_completed += 1
            i += 1
    return        


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
    for year in [2016, 2017, 2018]:
        parameters['year'] = year
        add_job(parameters)

    print('Job list created')

    # run jobs

    job = 0 
    if len(sys.argv) == 3:
        job = int(sys.argv[2])

    if job >= num_jobs:
        sys.exit()

if __name__ == "__main__":
    error_handling()
    main()