#!/bin/sh


# Parameters

    ########## Generic ##########

    year=2018
    num_iterations=5000

    ######## Directories ########

    demand_file="../demand_2018/WESTERN_IC.csv"
    eia_folder="../eia8602018/"
    solar_file="/scratch/mtcraig_root/mtcraig1/shared_data/merraData/cfs/wecc/2018_solar_ac_generation.nc"
    wind_file="/scratch/mtcraig_root/mtcraig1/shared_data/merraData/cfs/wecc/2018_wind_ac_generation.nc"

    ########## System ###########

    system_setting="system-0-saved.npy" #none, save, or load
    balancing_authority="0" #otherwise set 0   
    nerc_region="WECC" # DO NOT identify both a balancing authority and a nerc region
    conventional_efor=.05
    vg_efor=1.0
    derate_conventional="True"
    oldest_year_manual=1950 #set to 0 if not known, used for removing generators

    ######## Generator ##########

    generator_type="solar"
    generator_capacity=100 #MW
    generator_latitude=47.75
    generator_longitude=-122.45
    generator_efor=.05

function update_vars {
    a=$year
    b=$num_iterations
    c=$demand_file
    d=$eia_folder
    e=$solar_file
    f=$wind_file
    g=$system_setting
    h=$balancing_authority
    i=$nerc_region
    j=$conventional_efor
    k=$vg_efor
    l=$derate_conventional
    m=$oldest_year_manual
    n=$generator_type
    o=$generator_capacity
    p=$generator_latitude
    q=$generator_longitude
    r=$generator_efor
}

function run_sim {
    update_vars
    python -u ijbd_elcc.py $a $b $c $d $e $f $g $h $i $j $k $l $m $n $o $p $q $r
}


################ DO NOT CHANGE ABOVE THIS LINE ############

echo "Seattle Solar Contributing to WECC 2018:"
echo " "

echo "1 MW:"  
    generator_capacity=1
    run_sim
echo " "

echo "10 MW:"  
    generator_capacity=10
    run_sim
echo " "

echo "100 MW:"  
    generator_capacity=100
    run_sim
echo " "

echo "1 GW:"  
    generator_capacity=1000
    run_sim
echo " "

echo "10 GW:"  
    generator_capacity=10000
    run_sim
echo " "

echo "100 GW:"  
    generator_capacity=100000
    run_sim
echo " "


echo "ijbd_elcc.sh complete"
    