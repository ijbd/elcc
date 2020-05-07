ELCC
===============


Isaac Bromley-Dulfano
University of Michigan ASSET Lab

Computational method for calculating effective load carrying capability (ELCC) of a solar or wind plant in the U.S. 

Instructions:
-------------

1. Download this repository via commmand line- 

    `git clone https://github.com/ijbd/elcc.git`

2. Go to bin-

    `cd elcc/bin/`

4. Change parameters in ijbd_elcc.sh

3. Change permission for shell script

    `chmod a+rx ijbd_elcc.sh`

5. Run in foreground

    `bash ijbd_elcc.sh`

6. Run in background 

    `nohup bash ijbd_elcc.sh > filename.out &`
    
Citations:
----------

[1] The Modern-Era Retrospective Analysis for Research and Applications, Version 2 (MERRA-2), Ronald Gelaro, et al., 2017, J. Clim., doi: 10.1175/JCLI-D-16-0758.1

[2] U.S. Energy Information Administration Form EIA-860 (2019)

[3] Tyler Ruggles, & David Farnham. (2019). EIA Cleaned Hourly Electricity Demand Data (Version v1.0_23Oct2019) [Data set]. Zenodo. http://doi.org/10.5281/zenodo.3517197

[4] William F. Holmgren, Clifford W. Hansen, and Mark A. Mikofski. “pvlib python: a python package for modeling solar energy systems.” Journal of Open Source Software, 3(29), 884, (2018). https://doi.org/10.21105/joss.00884

[5] Blair, Nate, Nicholas DiOrio, Janine Freeman, Paul Gilman, Steven Janzou, Ty Neises, and Michael Wagner. 2018. System Advisor Model (SAM) General Description (Version 2017.9.5). Golden, CO: National Renewable Energy Laboratory. NREL/ TP-6A20-70414. https://www.nrel.gov/docs/fy18osti/70414.pdf

