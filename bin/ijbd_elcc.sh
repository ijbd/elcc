#!/bin/sh


# Parameters

    ########## Generic ##########

    year=2018
    num_iterations=1000

    ######## Directories ########

    demand_file="../demand/PACE.csv"
    eia_folder="../eia8602018/"
    solar_file="../wecc_powGen/2018_solar_ac_generation.nc"
    wind_file="../wecc_powGen/2018_wind_ac_generation.nc"

    ########## System ###########

    system_setting="none" #none, save (save processed system), or filename (of saved processed system)

    balancing_authority="PACE" #otherwise set 0   
    #or
    nerc_region="0" # DO NOT identify both a balancing authority and a nerc region

    conventional_efor=.05
    vg_efor=1.0
    derate_conventional="True"
    oldest_year_manual=1950 #set to 0 if not known, used for removing generators

    ######## Generator ##########

    generator_type="solar" #solar or wind
    generator_capacity=100 #MW
    generator_latitude=41
    generator_longitude=-112
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

echo "Salt Lake City, UT 100 MW Solar contributing to PACE system:"
echo " "


run_sim

echo " "
echo "ijbd_elcc.sh complete"
    