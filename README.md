ELCC Calculator
===============

Computational method for calculating effective load carrying capability (ELCC) of a solar or wind plant in the U.S. Developed by the University of Michigan Advancing Sustainable Systems through low-impact Energy Technologies (ASSET) Lab.

Instructions:
-------------

1. Download this repository via commmand line- 

    `git clone https://github.com/ijbd/elcc.git`


2. Run ELCC calculation from bin via command line

    `python elcc_driver.py`

3. Change parameters with key-value pairs

    `python elcc_driver.py year 2016 region CISO`

4. See a full list of parameters in elcc_driver.py

5. For multi-worded parameters use underscores

    `python elcc_driver.py conventional_efor .1`

6. Changing the root directory will create a unique folder based on the arguments and redirect output to a file

    `python elcc_driver.py root_directory testing/ region CISO`

7. To add a balancing authority for a simulation. Use Tyler Ruggles' cleaned EIA-860 data from GitHub. Place it in the demand folder with the capitalized abbreviation for that balancing authority

8. To use, ARC-TS launcher to calculate ELCC values synchronously, refer to 9.

9. This manual is incomplete, but I'm happy to help if you're having trouble with anything! Email me at ijbd@umich.edu

`src/` File Overview:
----------
`elcc_impl.py`: Contains the actual ELCC calculation, including file I/O, state sampling Monte Carlo simulation, storage operation, etc.

`elcc_driver.py`: Contains and fills dictionaries of parameters needed for the elcc calculation. At bare minimum to run an ELCC calculation, this is the script that must be run.

`elcc.py`: Poorly-named. Used for running a single ELCC calculations with SLURM. For calculations with many iterations (>1000), this is necessary as greatlakes will halt jobs not running on SLURM after a few hours. `argv[1]` should be a valid output directory. 

`elcc_single_job.sbat`: Contains the SLURM resource allocations associated with the `elcc.py`

`elcc_map_base_cases.py`: For generating ELCC maps. The current file is set up for our particular study, so a fair bit of adjustment needs to be done. **CAUTION:** This script will start off potentially thousands of SLURM jobs at one time incurring a not insignificant charge! 

`elcc_batch_job.sbat`: Contains the SLURM resource allocations associated with the `elcc.py`. The difference between `elcc_batch_job.sbat` and `elcc_single_job.sbat` are the number of tasks per node. Which can be optimized to minimize the charge incurred by SLURM. 

`extract_results.py`: Used for extracting all ELCC values for a batch job (i.e. map). In the future, should automatically be run, but for now it's manual.

**NOTE:** Running maps of ELCCs is a little messy, but there are a few really important steps that need to be followed. If you need to run an ELCC map with SLURM. Please reach out, and we should schedule a meeting so I can explain it better.

Citations:
----------

[1] The Modern-Era Retrospective Analysis for Research and Applications, Version 2 (MERRA-2), Ronald Gelaro, et al., 2017, J. Clim., doi: 10.1175/JCLI-D-16-0758.1

[2] U.S. Energy Information Administration Form EIA-860 (2019)

[3] Tyler Ruggles, & David Farnham. (2019). EIA Cleaned Hourly Electricity Demand Data (Version v1.0_23Oct2019) [Data set]. Zenodo. http://doi.org/10.5281/zenodo.3517197

[4] William F. Holmgren, Clifford W. Hansen, and Mark A. Mikofski. “pvlib python: a python package for modeling solar energy systems.” Journal of Open Source Software, 3(29), 884, (2018). https://doi.org/10.21105/joss.00884

[5] Blair, Nate, Nicholas DiOrio, Janine Freeman, Paul Gilman, Steven Janzou, Ty Neises, and Michael Wagner. 2018. System Advisor Model (SAM) General Description (Version 2017.9.5). Golden, CO: National Renewable Energy Laboratory. NREL/ TP-6A20-70414. https://www.nrel.gov/docs/fy18osti/70414.pdf

[6] de Chalendar, Jacques A., John Taggart, and Sally M. Benson. "Tracking emissions in the US electricity system." Proceedings of the National Academy of Sciences 116.51 (2019): 25497-25502.

*citations not up-to-date*