import numpy as np
import sys
from netCDF4 import Dataset

year = sys.argv[1] 
gen_type = sys.argv[2]

data = Dataset(year+'_'+gen_type+'_generation_cf.nc')

cf = np.array(data.variables['cf'])

print('Maximum:',np.max(cf))
print('Minimum:',np.min(cf))
print('Average:',np.average(cf))
