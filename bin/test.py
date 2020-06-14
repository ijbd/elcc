import storage
import numpy as np
from elcc_impl import get_demand_data

def main():
    storage_fleet = storage.make_storage(10000,1000,1000,.8,0)
    #storage_fleet = storage.get_storage_fleet('../eia8602018/','WECC',2018,.8)
    hourly_load = get_demand_data('../demand/PACE.csv',2018)
    hourly_capacity = np.ones(8760).reshape(8760,1)*5500
    storage.get_hourly_storage_contribution(1,hourly_capacity,hourly_load,storage_fleet)

main()