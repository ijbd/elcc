import os

root_directory = '/scratch/mtcraig_root/mtcraig1/shared_data/elccJobs/tests/lowResolutionMatrixTest/'

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
    
    os.system('sbat elcc_batch_job.sbat' + parameter_string)

def main():

    parameters = dict()

    # universal parameters
    parameters['nameplate'] = 100
    parameters['generator_type'] = 'solar'

    # variables

    locations = dict()
    locations['name'] =     ['Seattle', 'Missoula', 'Miles City',   'Fort Collins', 'Albuquerque',  'Phoenix',  'Wendover', 'Los Angeles',  'Sacramento']
    locations['latitude'] = [47.6,      46.9,       46.4,           40.6,           35.1,           33.4,       40.7,       34.1,           38.6]
    locations['longitude'] = [-122.3,    -114.0,    -105.8,          -105.1,         -106.7,         -112.1,     -114.1,     -118.2,         -121.5]

    years = [2016, 2017, 2018]

    regions = ['PACE','CISO','WECC']

    for year in years:
        parameters['year'] = year
        for region in regions:
            parameters['region'] = region
            for location in range(len(locations['name'])):
                parameters['latitude'] = locations['latitude'][location]
                parameters['longitude'] = locations['longitude'][location]

                print('Running:',year,region,locations['name'][location])
                run_job(parameters)

if __name__ == "__main__":
    error_handling()
    main()