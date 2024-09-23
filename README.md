# cmod_simple_fitting
Some simple routines for fitting Thomson Scattering profile data on Alcator C-Mod

The script run_multiple_shots.py takes in a list of C-Mod shots, and fits all the Thomson profiles in each shot.

The fits are returned in a dictionary (saved as a .pickle file).

## Example Sub-Heading

### Example sub-sub-heading


# TODO:
- [ ] Add in Ly-alpha analysis capabilities
- [ ] Implement option for post-fit outlier rejection + refit in the 1D analysis
- [x] Fix the scaling to TCI data in the 1D fits
- [x] Implement a Monte-Carlo method for error bars in the 1D fitting routine
- [ ] Let the 2D fit function also fit a cubic function if it wants
- [x] Split the functions into 4 or 5 separate files to make them more manageable
- [x] Add in the quadratic function option for smoothing
- [ ] Clean up some of the long functions?
- [ ] Add in options to do a combined fit for multiple different shots/times
- [ ] Add in option to set a floor on ne and Te in the 2D fitting routines. Currently, some very small negative ne values in the SOL are causing a lot of things to break

