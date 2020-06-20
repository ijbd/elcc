import numpy as np 
import pandas as pd 
from elcc_impl import get_hourly_load
from storage_impl import make_storage, get_hourly_storage_contribution, get_storage_fleet, append_storage

def main():
    num_iterations = 2
    year = 2018
    capacity = 5800
    storage_unit1 = make_storage(True, 100,100,100,.8,0,0,False)
    storage_unit2 = make_storage(False, 100,100,100,.8,0,0,False)
    storage_unit = append_storage(storage_unit1,storage_unit2)
    #storage_unit = get_storage_fleet(True,'../eia8602018/','WECC',2018,.8,0,0,False)
    hourly_load = get_hourly_load('../demand/PACE.csv',year)
    hourly_capacity = np.ones((8760,num_iterations))*capacity
    hourly_storage_contribution = get_hourly_storage_contribution(2,hourly_capacity,hourly_load,storage_unit)
    hourly_storage_contribution_b = get_hourly_storage_contribution(2,hourly_capacity,hourly_load,storage_unit)

    data=np.array([ hourly_load,
                    hourly_capacity[:,0],
                    hourly_load-hourly_capacity[:,0],
                    hourly_storage_contribution[:,0],
                    hourly_storage_contribution[:,1],
                    hourly_storage_contribution_b[:,0],
                    hourly_storage_contribution_b[:,1]]).T

    df = pd.DataFrame(  data=data,
                        columns=[   "Load (MW)",
                                    "Capacity (MW)",
                                    "Net Load (MW)",
                                    "Storage (Itr 1)", 
                                    "Storage (Itr 2)",
                                    "Storage (Itr 3)",
                                    "Storage (Itr 4)"])
    
    df.to_csv("storage_dbg.csv")

if __name__ == "__main__":
    main()
    pass