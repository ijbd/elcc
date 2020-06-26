import numpy as np 
import pandas as pd 
import os

root_directory = "../../archive/2018_PACE_storage_high_res/"
key_words = ['supplemental storage','supplemental storage power capacity','supplemental storage energy capacity','Capacity removed']
key_words.append('ELCC')

def get_results(filename, key_words):
    results = dict()

    with open(filename,'r') as f:
        for line in f:
            for key in key_words:
                query = key+' : '
                if line.find(query) != -1:
                    print(line.find(query))
                    results[key] = line[line.find(query)+len(query):-1]
    return results



def main():
    # find desired output files for all jobs in batch
    printout_files = []

    for root, dirs, files in os.walk(root_directory):
        for filename in files:
            if filename.endswith("print.out"):
                printout_files.append(os.path.join(root, filename))
    
    all_results = pd.DataFrame()

    # go through file by file and extract key words
    for filename in printout_files:
        job_results = get_results(filename, key_words)
        print(job_results)
        all_results = all_results.append(job_results,ignore_index=True)

    print(all_results.head)
    all_results.to_csv(root_directory+'results.csv')

if __name__ == "__main__":
    main()
    pass