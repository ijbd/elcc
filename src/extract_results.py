import numpy as np 
import pandas as pd 
import os
import sys

root_directory = sys.argv[1]
title = sys.argv[2]

# save mistakes
if root_directory[-1] != '/': root_directory = root_directory + '/'

key_words = sys.argv[3:]

def get_results(filename, key_words):
    results = dict()

    with open(filename,'r') as f:
        for line in f:
            for key in key_words:
                query = key+' : '
                if line.find(query) != -1:
                    if key not in results:
                        results[key] = line[line.find(query)+len(query):-1]
                    
    return results

def main():
    # find desired output files for all jobs in batch
    printout_files = []

    for root, dirs, files in os.walk(root_directory):
        for filename in files:
            if filename.endswith("print.out"):
                printout_files.append(os.path.join(root, filename))
    
    print(len(printout_files),"jobs found")

    all_results = pd.DataFrame()

    # go through file by file and extract key words
    for filename in printout_files:
        job_results = get_results(filename, key_words)
        all_results = all_results.append(job_results,ignore_index=True)

    all_results.to_csv(root_directory+'results.csv')
    all_results.to_csv(title+'_results.csv')

if __name__ == "__main__":
    main()
    pass