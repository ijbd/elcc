import os, sys

region = sys.argv[1]
year = sys.argv[2]

# 1 GW solar
os.system('python extract_results.py ../../elccJobs/'+region+'/'+str(year)+'/1GWsolar 1_GW_Solar ELCC latitude longitude')

# 1 GW wind 
os.system('python extract_results.py ../../elccJobs/'+region+'/'+str(year)+'/1GWwind 1_GW_Wind ELCC latitude longitude')

# 100 MW solar
os.system('python extract_results.py ../../elccJobs/'+region+'/'+str(year)+'/100MWsolar 100_MW_Solar ELCC latitude longitude')

# 100 MW wind
os.system('python extract_results.py ../../elccJobs/'+region+'/'+str(year)+'/100MWwind 100_MW_Wind ELCC latitude longitude')

# 5 GW solar
os.system('python extract_results.py ../../elccJobs/'+region+'/'+str(year)+'/5GWsolar 5_GW_Solar ELCC latitude longitude')

# 5 GW wind 
os.system('python extract_results.py ../../elccJobs/'+region+'/'+str(year)+'/5GWwind 5_GW_Wind ELCC latitude longitude')

# 1 GW solar 2x renewables
os.system('python extract_results.py ../../elccJobs/'+region+'/'+str(year)+'/1GWsolar2xRenewables 1_GW_Solar_2x_Renewables ELCC latitude longitude')

# 1 GW wind 2x renewables
os.system('python extract_results.py ../../elccJobs/'+region+'/'+str(year)+'/1GWwind2xRenewables 1_GW_Wind_2x_Renewables ELCC latitude longitude')

# 1 GW solar + storage
os.system('python extract_results.py ../../elccJobs/'+region+'/'+str(year)+'/1GWsolar500MWstorage1hour 1_GW_Solar_500_MW_1_Hour_Storage ELCC latitude longitude')

# 1 GW wind + storage
os.system('python extract_results.py ../../elccJobs/'+region+'/'+str(year)+'/1GWwind500MWstorage1hour 1_GW_Wind_500_MW_1_Hour_Storage ELCC latitude longitude')
