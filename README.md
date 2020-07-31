ELCC Calculator
===============

Computational method for calculating effective load carrying capability (ELCC) of a solar or wind plant in the U.S. Developed by the University of Michigan Advancing Sustainable Systems through low-impact Energy Technologies (ASSET) Lab.

Instructions:
-------------

1. Download this repository via commmand line- 

    `git clone https://github.com/ijbd/elcc.git`


2. Run simulation from bin in foreground from command-line

    `cd bin/`
    `python elcc_master.py`

3. Change parameters via key-value pairs on command line

    `python elcc_master.py year 2016 region CISO`

4. See a full list of parameters in elcc_master.py

5. For multi-worded parameters use underscores

    `python elcc_master.py conventional_efor .1`

6. Changing the root directory will create a unique folder in the root directory based on the arguments and redirect output

    `python elcc_master.py root_directory testing/ region CISO`

7. To add a region to simulation. Use Tyler Ruggles' cleaned EIA-860 data from GitHub. Place it in the demand folder with the capitalized abbreviation for that balancing authority

8. To use, ARC-TS launcher use elcc.py, refer to 9.

9. This manual is incomplete, but I'm happy to help if you're having trouble with anything! Email me at ijbd@umich.edu

Citations:
----------

[1] The Modern-Era Retrospective Analysis for Research and Applications, Version 2 (MERRA-2), Ronald Gelaro, et al., 2017, J. Clim., doi: 10.1175/JCLI-D-16-0758.1

[2] U.S. Energy Information Administration Form EIA-860 (2019)

[3] Tyler Ruggles, & David Farnham. (2019). EIA Cleaned Hourly Electricity Demand Data (Version v1.0_23Oct2019) [Data set]. Zenodo. http://doi.org/10.5281/zenodo.3517197

[4] William F. Holmgren, Clifford W. Hansen, and Mark A. Mikofski. “pvlib python: a python package for modeling solar energy systems.” Journal of Open Source Software, 3(29), 884, (2018). https://doi.org/10.21105/joss.00884

[5] Blair, Nate, Nicholas DiOrio, Janine Freeman, Paul Gilman, Steven Janzou, Ty Neises, and Michael Wagner. 2018. System Advisor Model (SAM) General Description (Version 2017.9.5). Golden, CO: National Renewable Energy Laboratory. NREL/ TP-6A20-70414. https://www.nrel.gov/docs/fy18osti/70414.pdf

[6] de Chalendar, Jacques A., John Taggart, and Sally M. Benson. "Tracking emissions in the US electricity system." Proceedings of the National Academy of Sciences 116.51 (2019): 25497-25502.

*citations not up-to-date*