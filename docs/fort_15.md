
## Run Parameters

Full List of ADCIRC Fort.15 Control Parameters for a 2D run

| Parameter | Description |
|-----------|-------------|
| [RUNDES](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#RUNDES) | Alpha-numeric run description (<=32 characters). |
| [RUNID](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#RUNID) | Alphanumeric run description (<=24 characters). |
| [NFOVER](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NFOVER)) | Non-fatal error override option. |
| | = 0: Inconsistent input parameters cause program termination. |
| | = 1: Inconsistent input parameters automatically corrected (when possible) to a default or consistent value. Note: Some parameters may still cause fatal errors and program termination. |
| [NABOUT](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NABOUT) | Logging level for ADCIRC output to screen and log file. |
| | DEBUG: Debug level and higher (for developers). |
| | ECHO: Echo level and higher (includes printing of input files). |
| | INFO: Info level and higher (important non-issue information). |
| | WARNING: Warning level and higher (non-fatal problems). |
| | ERROR: Error level only (severe problems). |
| [NSCREEN](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NSCREEN) | Controls log message output to the screen. |
| | <0: Writes to adcirc.log file. |
| | 0: No log messages on screen. |
| | >0: Log messages on screen (standard output). |
| [IHOT](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#IHOT)| Model hot start option. |
| | 0: Cold start the model. |
| | 17: Hot start from ASCII file fort.17. |
| | 67: Hot start model using input information in hot start file fort.67. |
| | 68: Hot start model using input information in hot start file fort.68. |
| | 367: Hot start model using input information in netCDF hot start file fort.67.nc. |
| | 368: Hot start model using input information in netCDF hot start file fort.68.nc. |
| | 567: Hot start model using input information in netCDF4 hot start file fort.67.nc. |
| | 568: Hot start model using input information in netCDF4 hot start file fort.68.nc. |
| [ICS](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#ICS)| Model coordinate system. |
| | 1: ADCIRC governing equations are in Cartesian coordinates. |
| | 2: ADCIRC governing equations are in spherical coordinates transformed into Cartesian coordinates. |
| [IM](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#IM)| Model type. |
| | 0: Barotropic 2DDI run using New GWCE and Momentum equation formulations. |
| | 1: Barotropic 3D run using New GWCE and velocity-based Momentum equations. |
| | 21: Baroclinic 3D run using New GWCE and velocity-based Momentum equations. |
| | 111112: Barotropic 2DDI run using the lumped GWCE (explicit mode). |
| | 611112: Barotropic 3D run using the lumped GWCE (explicit mode). |
| [IDEN](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#IDEN)| Form of density forcing in a 3D run. |
| | -4: Diagnostic Baroclinic ADCIRC run with Salinity and Temperature forcing. |
| | -3: Diagnostic Baroclinic ADCIRC run with Temperature forcing. |
| | -2: Diagnostic Baroclinic ADCIRC run with Salinity forcing. |
| | -1: Diagnostic Baroclinic ADCIRC run with Sigma T forcing. |
| | 0: Barotropic model run. |
| | 1: Prognostic Baroclinic ADCIRC run with Sigma T forcing. |
| | 2: Prognostic Baroclinic ADCIRC run with Salinity forcing. |
| | 3: Prognostic Baroclinic ADCIRC run with Temperature forcing. |
| | 4: Prognostic Baroclinic ADCIRC run with Salinity and Temperature forcing. |
| [NOLIBF](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NOLIBF)| Bottom stress parameterization in a 2DDI ADCIRC run. |
| | 0: Linear bottom friction law (FFACTOR specified below). |
| | 1: Quadratic bottom friction law (FFACTOR specified below). |
| | 2: Hybrid nonlinear bottom friction law (FFACTOR determined by specified parameters). |
| [NOLIFA](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NOLIFA)| Finite amplitude terms in ADCIRC. |
| | 0: Finite amplitude terms NOT included in the model run. |
| | 1: Finite amplitude terms included in the model run. |
| | 2: Finite amplitude terms included in the model run with wetting and drying of elements enabled. |
| [NOLICA](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NOLICA)| Advective terms in ADCIRC. |
| | 0: Advective terms NOT included in the computations. |
| | 1: Advective terms included in the computations. |
| [NOLICAT](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NOLICAT)| Time derivative portion of the advective terms in the GWCE continuity equation. |
| | 0: Time derivative portion of the advective terms NOT included in the computations. |
| | 1: Time derivative portion of the advective terms included in the computations. |
| [NWP](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NWP)| Number of nodal attributes used in the run. |
| [NCOR](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NCOR)| Parameter controlling the Coriolis parameter. |
| | = 0, spatially constant Coriolis parameter. |
| | = 1, spatially variable Coriolis parameter. |
| [NTIP](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NTIP)| Parameter controlling tidal forcings. |
| | = 0, no tidal forcings used. |
| | = 1, tidal potential forcing used. |
| | = 2, tidal potential & self attraction/load tide forcings used. |
| [NWS](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NWS)| Parameter controlling whether wind velocity or stress, wave radiation stress and atmospheric pressure are used to force ADCIRC. |
|  | 0 -  No wind, radiation stress, or atmospheric pressure forcings are used.                                       |
|  | 1 -  Wind stress and atmospheric pressure are read in at all grid nodes from a meteorological forcing input file. |
|  | 2 -  Wind stress and atmospheric pressure are read in at specified time intervals from a meteorological file.    |
|  | -2 -  Wind stress and atmospheric pressure are read in at specified time intervals assuming a hot start.          |
|  | 3 -  Wind velocity is read in from a wind file in US Navy Fleet Numeric format.                                  |
|  | 4 -  Wind velocity and atmospheric pressure are read in at selected grid nodes from a meteorological file.       |
|  | -4  -  Wind velocity and atmospheric pressure are read in at selected grid nodes assuming a hot start.             |
|  | 5 -  Wind velocity and atmospheric pressure are read in at all grid nodes from a meteorological file.            |
|  | -5  -  Wind velocity and atmospheric pressure are read in at all grid nodes assuming a hot start.                  |
|  | 6 -  Wind velocity and atmospheric pressure are read in for a rectangular grid and interpolated onto ADCIRC grid.|
|  | 7   -  Surface stress and pressure values are read in on a regular grid from a meteorological file.                |
|  | -7  -  Surface stress and pressure values are read in on a regular grid assuming a hot start.                      |
|  | 8 -  Hurricane parameters are read in and wind velocity and pressure are calculated internally by ADCIRC.       |
|  | 9 -  Asymmetric hurricane model (no longer supported).                                                          |
|  | 10 -  Wind velocity and atmospheric pressure are read in from National Weather Service (NWS) Aviation model files.|
|  | 11 -  Wind velocity and atmospheric pressure are read in from NWS ETA 29km model files.                           |
|  | 12 -  Wind velocity and atmospheric pressure are provided in the OWI format on rectangular grids.                |
|  | 15 -  HWind files are data assimilated snapshots of wind velocity fields of tropical cyclones.                    |
|  | 19 -  User-defined selection of isotachs and modification of RMAX and Holland's B parameter.                      |
| [NRAMP](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#AMP) | Ramp option parameter controlling whether a ramp is applied to ADCIRC forcing functions. |
| | = 0 No ramp function is used with forcing functions; full strength forcing is applied immediately upon cold start. |
| | = 1 A single hyperbolic tangent ramp function of specified duration (DRAMP, in days relative to the cold start time) will be applied to all forcing. See description of the DRAMP line for further information on the ramp function. |
| | = 2 Same as NRAMP=1, except that a second, separate hyperbolic tangent ramp of specified duration (DRAMPExtFlux, in days relative to cold start time plus FluxSettlingTime) specifically for external flux forcing (e.g., river boundary conditions) will also be read on the DRAMP line. In addition, the FluxSettlingTime parameter for IBTYPE=52 river boundaries will also be specified on the DRAMP line. If there are no IBTYPE=52 boundaries in the mesh (fort.14) file, the FluxSettlingTime will be read but ignored. See description of DRAMP for further information. |
| | = 3 Same as NRAMP=2, except that a third, separate hyperbolic tangent ramp of specified duration (DRAMPIntFlux, in days relative to cold start time plus FluxSettlingTime) specifically for internal flux forcing (e.g., flows over levees and under culverts) will also be read on the DRAMP line. See the description of the DRAMP line for further information. |
| | = 4 Same as NRAMP=3, except that a fourth, separate hyperbolic tangent ramp of specified duration (DRAMPElev, in days relative to cold start time plus FluxSettlingTime) specifically for elevation specified boundary forcing (e.g., tidal boundaries) will also be read on the DRAMP line. See the description of the DRAMP line for further information. |
| | = 5 Same as NRAMP=4, except that a fifth, separate hyperbolic tangent ramp of specified duration (DRAMPTip, in days relative to cold start time plus FluxSettlingTime) specifically for tidal potential forcing will also be read on the DRAMP line. See the description of the DRAMP line for further information. |
| | = 6 Same as NRAMP=5, except that a sixth, separate hyperbolic tangent ramp of specified duration (DRAMPMete, in days relative to cold start time plus FluxSettlingTime) specifically for meteorological forcing (i.e., wind and atmospheric pressure) will also be read on the DRAMP line. See the description of the DRAMP line for further information. |
| | = 7 Same as NRAMP=6, except that a seventh, separate hyperbolic tangent ramp of specified duration (DRAMPWRad, in days relative to cold start time plus FluxSettlingTime) specifically for wave radiation stress forcing will also be read on the DRAMP line. See the description of the DRAMP line for further information. |
| | = 8 Same as NRAMP=7, except that a delay parameter (DUnRampMete, in days relative to cold start time plus FluxSettlingTime) will also be read from the DRAMP line. The meteorological ramp delay parameter DUnRampMete is useful in cases where a meteorologically-forced run will be hotstarted from a long term meteorologically-free tidal spinup from cold start. The meteorological
| [G](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#G) | Gravitational constant determining ADCIRC's distance units; required values when ICS = 2, NTIP = 1, or NCOR = 1: G = 9.81 m/sec². |
| [TAU0](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#TAU0) | Weighting factor for the Generalized Wave-Continuity Equation (GWCE), influencing primitive and wave contributions; various options for TAU0 values based on conditions and calculations. |
| | = 0: GWCE as a pure wave equation. |
| | < 1: GWCE behaves like a pure primitive continuity equation. Suggested TAU0 range: 0.005 - 0.1. |
| | = -1: TAU0 varies spatially, constant in time; calculated based on depth. |
| | = -2: TAU0 varies spatially, constant in time; calculated based on depth. |
| | = -3: TAU0 varies spatially and in time; computed from TAU0Base and TK(i). |
| | = -5: FullDomainTimeVaryingTau0 = .True.; spatially and time-varying TAU0 dependent on local friction, limited by Tau0FullDomainMin and Tau0FullDomainMax. |
| [DTDP](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#DTDP) | ADCIRC time step in seconds; predictor-corrector algorithm usage based on value. |
| [STATIM](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#STATIM) | Starting simulation time in days; nonzero value aligns model output times with a specific reference. |
| [REFTIM](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#REFTIM) | Reference time in days; used for harmonic forcing and analysis terms computation. |
| [WTIMINC](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#WTIMINC) | Time increment between meteorological forcing data sets in seconds; dependent on NWS parameter (see Supplemental Meteorological/Wave/Ice Parameters table). |
| [DRAMP](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#DRAMP) | Value (in decimal days) for ramping up ADCIRC forcings. |
| [DRAMPExtFlux](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#DRAMPExtFlux) | Value (in decimal days) for ramping up nonzero external flux boundary condition. |
| [FluxSettlingTime](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#FluxSettlingTime) | Time in days for river flux boundary condition and bottom friction to equilibrate. |
| | Notes: Until FluxSettlingTime elapses, only external boundary flux forcing is active. Afterward, other forcings start ramping up. Non-periodic flux conditions not supported with IBTYPE=52. |
| [DRAMPIntFlux](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#DRAMPIntFlux) | Value (in decimal days) for ramping up nonzero internal flux boundary condition. |
| [DRAMPElev](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#DRAMPElev) | Value (in decimal days) for ramping up elevation-specified boundary condition. |
| [DRAMPTip](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#DRAMPTip) | Value (in decimal days) for ramping up tidal potential. |
| [DRAMPMete](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#DRAMPMete) | Value (in decimal days) for ramping up wind and atmospheric pressure. |
| [DRAMPWRad](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#DRAMPWRad) | Value (in decimal days) for ramping up wave radiation stress. |
| [DUnRampMete](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#DUnRampMete) | Meteorological ramp delay parameter (in decimal days) relative to ADCIRC cold start time. | 
| [A00, B00, C00](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#A00B00C00) | Time weighting factors (at time levels k+1, k, k-1, respectively) in the GWCE. |
| [H0](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#H0) | Minimum water depth. |
| | If NOLIFA = 0, H0 = minimum bathymetric depth. |
| | If NOLIFA = 2, H0 = nominal water depth for a node to be considered dry. |
| [INTEGER](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#INTEGER) | No longer needed, values will be ignored. |
| [VELMIN](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#VELMIN) | Minimum velocity for wetting. |
| [SLAM0](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#SLAM0), [SFEA0](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#SFEA0) | Longitude and latitude for the CPP coordinate projection center. |
| [TAU](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#TAU) | Linear friction coefficient for bottom friction. |
| [CF](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#CF) | 2DDI quadratic bottom friction coefficient. |
| [HBREAK](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#HBREAK) | Break depth utilized for NOLIBF = 2. |
| [FTHETA](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#FTHETA) | Parameter for hybrid bottom friction relationship (NOLIBF = 2). |
| [FGAMMA](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#FGAMMA) | Parameter for hybrid bottom friction relationship (NOLIBF = 2). |
| [ESLM](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#ESLM) | Spatially constant horizontal eddy viscosity for momentum equations. |
| [ESLC](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#ESLC) | Spatially constant horizontal eddy diffusivity for transport equation (only if IM = 10). |
| [CORI](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#CORI) | Constant Coriolis coefficient (used when NCOR = 0). |
| [NBFR](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NBFR) | Number of periodic forcing frequencies on elevation specified boundaries. |
| | If NBFR=0, elevation boundary condition is assumed to be non-periodic. |
| [BOUNTAG(k)](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#BOUNTAG) | Alphanumeric descriptor for forcing frequency, nodal factor, and equilibrium argument for tidal forcing on elevation specified boundaries. |
| [AMIG(k), FF(k), FACE(k)](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#AMIG) | Forcing frequency, nodal factor, and equilibrium argument for tidal forcing on elevation specified boundaries. |
| [ALPHA](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#ALPHA) | Alphanumeric descriptor for amplitude and phase of harmonic forcing function at elevation specified boundaries. |
| [EMO(k,j), EFA(k,j)](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#EMO) | Amplitude and phase of harmonic forcing function at elevation specified boundaries for frequency k and node j. |
| [ANGINN](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#ANGINN) | Inner angle threshold for flow boundary nodes with normal flow essential boundary condition. |
| [NFFR](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NFFR) | Number of frequencies in specified normal flow external boundary condition. |
| | If NFFR=0 or NFFR=-1, the normal flow boundary condition is assumed to be non-periodic. |
| [FBOUNTAG(k)](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#FBOUNTAG) | Alphanumeric descriptor for forcing frequency, nodal factor, and equilibrium argument for periodic normal flow forcing on flow boundaries. |
| [FAMIGT(k), FFF(k), FFACE(k)](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#FAMIGT) | Forcing frequency, nodal factor, and equilibrium argument for periodic normal flow forcing on flow boundaries. |
| [ALPHA](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#ALPHA) | Alphanumeric descriptor for amplitude and phase of periodic normal flow/unit width for frequency k and boundary node j. |
| [QNAM(k,j), QNPH(k,j)](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#QNAM) | Amplitude and phase of periodic normal flow/unit width for frequency k and "specified normal flow" boundary node j. |
| [ENAM(k,j), ENPH(k,j)](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#ENAM) | Amplitude and phase of outgoing wave in IBTYPE=32 boundary condition. |
| [NOUTE](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NOUTE)      | Output parameters controlling the time series output for elevation solutions at selected elevation recording stations (fort.61 output) |
|            | =-3: Output in netCDF format, new fort.61.nc file created after hot start |
|            | =-2: Output in binary format, new fort.61 file created after hot start |
|            | =-1: Output in standard ascii format, new fort.61 file created after hot start |
|            | = 0: No output at elevation recording stations |
|            | = 1: Output in standard ascii format, continued output merged into existing fort.61 file after hot start |
|            | = 2: Output in binary format, continued output merged into existing fort.61 file after hot start |
|            | = 3: Output in netCDF format, continued output merged into existing fort.61.nc file after hot start |
| [TOUTSE](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#TOUTSE)     | Number of days after which elevation station data is recorded to fort.61 (relative to STATIM) |
| [TOUTFE](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#TOUTFE)     | Number of days after which elevation station data ceases to be recorded to fort.61 (relative to STATIM) |
| [NSPOOLE](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NSPOOLE)    | Number of time steps at which information is written to fort.61 |
| [NSTAE](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NSTAE)      | Number of elevation recording stations |
| [NOUTV](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NOUTV)      | Output parameters controlling the time series output for velocity solutions at selected velocity recording stations (fort.62 output) |
|            | =-3: Output in netCDF format, new fort.62.nc file created after hot start |
|            | =-2: Output in binary format, new fort.62 file created after hot start |
|            | =-1: Output in standard ascii format, new fort.62 file created after hot start |
|            | = 0: No output at velocity recording stations |
|            | = 1: Output in standard ascii format, continued output merged into existing fort.62 file after hot start |
|            | = 2: Output in binary format, continued output merged into existing fort.62 file after hot start |
|            | = 3: Output in netCDF format, continued output merged into existing fort.62.nc file after hot start |
| [TOUTSV](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#TOUTSV)     | Number of days after which velocity station data is recorded to fort.62 (relative to STATIM) |
| [TOUTFV](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#TOUTFV)     | Number of days after which velocity station data ceases to be recorded to fort.62 (relative to STATIM) |
| [NSPOOLV](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NSPOOLV)    | Number of time steps at which information is written to fort.62 |
| [NSTAV](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NSTAV)      | Number of velocity recording stations |
| [NOUTC](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NOUTC)      | Output parameters controlling the time series output for concentration solutions at selected concentration recording stations (fort.91 output) |
|            | =-2: Output in binary format, new fort.91 file created after hot start |
|            | =-1: Output in standard ascii format, new fort.91 file created after hot start |
|            | = 0: No output at concentration recording stations |
|            | = 1: Output in standard ascii format, continued output merged into existing fort.91 file after hot start |
|            | = 2: Output in binary format, continued output merged into existing fort.91 file after hot start |
| [TOUTSC](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#TOUTSC)     | Number of days after which concentration station data is recorded to fort.91 (relative to STATIM) |
| [TOUTFC](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#TOUTFC)     | Number of days after which concentration station data ceases to be recorded to fort.91 |
| [NSPOOLC](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NSPOOLC)    | The number of time steps at which information is written to fort.91; i.e. the output is written to fort.91 every NSPOOLC time steps after TOUTSC                                    |
| [NSTAC](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NSTAC)      | The number of concentration recording stations                                                                                                                                       |
| [NOUTM](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NOUTM)      | Output parameters which control the time series output provided for met data at selected met recording stations (units 71&72 output)                                                  |
|            | -3: Output is provided at the selected met recording stations in netCDF format. Following a hot start, new fort.71.nc&72.nc files are created.                                     |
|            | -2: Output is provided at the selected met recording stations in binary format. Following a hot start, new fort.71&72 files are created.                                             |
|            | -1: Output is provided at the selected met recording stations in standard ascii format. Following a hot start, new fort.71&72 files are created.                                   |
|            | 0: No output is provided at the selected met recording stations.                                                                                                                     |
|            | 1: Output is provided at the selected met recording stations in standard ascii format. Following a hot start, continued output is merged into the existing fort.71&72 files.        |
|            | 2: Output is provided at the selected met recording stations in binary format. Following a hot start, continued output is merged into the existing fort.71&72 files.                |
|            | 3: Output is provided at the selected met recording stations in netCDF format. Following a hot start, continued output is merged into the existing fort.71.nc&72.nc files.            |
| [TOUTSM](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#TOUTSM)     | The number of days after which met station data is recorded to units 71&72 (TOUTSM is relative to STATIM)                                                                          |
| [TOUTFM](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#TOUTFM)     | The number of days after which met station data ceases to be recorded to units 71&72 (is relative to STATIM)                                                                       |
| [NSPOOLM](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NSPOOLM)    | The number of time steps at which information is written to units 71&72; i.e., output is written to units 71&72 every NSPOOLM time steps after TOUTSM                                |
| [NSTAM](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NSTAM)      | The number of meteorological recording stations                                                                                                                                      |
| [NOUTGE](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NOUTGE)     | Output parameters which control the time series output provided for global elevation solutions at all nodes within the domain (fort.63 output)                                          |
|            | -3: Global elevation output is provided in netCDF format. Following a hot start, a new fort.63.nc file is created.                                                                  |
|            | -2: Global elevation output is provided in binary format. Following a hot start, a new fort.63 file is created.                                                                      |
|            | -1: Global elevation output is provided in standard ascii format. Following a hot start, a new fort.63 file is created.                                                               |
|            | 0: No global elevation output is provided.                                                                                                                                          |
|            | 1: Global elevation output is provided in standard ascii format. Following a hot start, continued output is merged into the existing fort.63 file.
| [NFREQ](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NFREQ)     | Number of frequencies for harmonic analysis (2DDI elevation and velocity)                                                                                                                                                                                                                                                                                                                 |
| [NAMEFR(k)](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NAMEFR) | Alphanumeric constituent descriptor (<= 10 characters)                                                                                                                                                                                                                                                                                                                                   |
| [HAFREQ(k)](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#HAFREQ) | Frequency (rad/s) for harmonic analysis                                                                                                                                                                                                                                                                                                                                                  |
| [HAFF(k)](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#HAFF)   | Nodal factor for harmonic analysis                                                                                                                                                                                                                                                                                                                                                        |
| [HAFACE(k)](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#HAFACE) | Equilibrium argument (degrees) for harmonic analysis                                                                                                                                                                                                                                                                                                                                      |
| [THAS](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#THAS)      | Start day for harmonic analysis (relative to STATIM)                                                                                                                                                                                                                                                                                                                                     |
| [THAF](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#THAF)      | End day for harmonic analysis (relative to STATIM)                                                                                                                                                                                                                                                                                                                                       |
| [NHAINC](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NHAINC)    | Time step interval for harmonic analysis                                                                                                                                                                                                                                                                                                                                                  |
| [FMV](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#FMV)       | Fraction of analysis period used for comparing elevation/velocity means and variances                                                                                                                                                                                                                                                                                                    |
| [NHASE](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NHASE)     | Perform harmonic analysis at elevation recording stations? (0/1)                                                                                                                                                                                                                                                                                                                         |
| [NHASV](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NHASV)     | Perform harmonic analysis at velocity recording stations? (0/1)                                                                                                                                                                                                                                                                                                                          |
| [NHAGE](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NHAGE)     | Perform harmonic analysis for global elevations? (0/1)                                                                                                                                                                                                                                                                                                                                   |
| [NHAGV](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NHAGV)     | Perform harmonic analysis for global velocities? (0/1)                                                                                                                                                                                                                                                                                                                                   |
| [NHSTAR](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NHSTAR)    | Generate hot start output files? (0/1/3/5)                                                                                                                                                                                                                                                                                                                                               |
| [NHSINC](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NHSINC)    | Time step interval for hot start output file generation                                                                                                                                                                                                                                                                                                                                   |
| [ITITER](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#ITITER)    | Solver type for GWCE (-1/1)                                                                                                                                                                                                                                                                                                                                                              |
| [ISLDIA](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#ISLDIA)    | Level of solver output detail (0-5)                                                                                                                                                                                                                                                                                                                                                      |
| [CONVCR](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#CONVCR)    | Convergence criteria for iterative solver                                                                                                                                                                                                                                                                                                                                                 |
| [ITMAX](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#ITMAX)     | Maximum number of iterations per time step                                                                                                                                                                                                                                                                                                                                                |
| [NCPROJ](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NCPROJ)   | Project Title (what is in the file).                                                                                                |
| [NCINST](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NCINST)   | Project Institution (where the file was produced).                                                                                  |
| [NCSOUR](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NCSOUR)   | Project Source (how it was produced, e.g., instrument type).                                                                         |
| [NCHIST](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NCHIST)   | Project History (audit trail of processing operations).                                                                             |
| [NCREF](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NCREF)     | Project References (pointers to publications, web documentation).                                                                    |
| [NCCOM](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NCCOM)     | Project Comments (any other comments about the file).                                                                                |
| [NCHOST](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NCHOST)   | Project Host.                                                                                                                      |
| [NCCONV](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NCCONV)   | Conventions.                                                                                                                        |
| [NCCONT](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NCCONT)   | Contact Information.                                                                                                               |
| [NCDATE](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NCDATE)   | The format of NCDATE must be as follows so that ADCIRC can create netcdf files that comply with the CF standard: yyyy-MM-dd hh:mm:ss tz. For example, if the cold start date/time of the run is midnight UTC on 1 May 2010, the NCDATE parameter should be set to 2010-05-01 00:00:00 UTC. |
| [WindDragLimit](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#WindDragLimit)           | Controls the ceiling on the wind drag coefficient.    |
| [DragLawString](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#DragLawString)           | Controls the formulation for calculating nodal wind drag coefficients.    |
| [rhoAir](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#rhoAir)                                      | Sets the density of air.                             |
| [waveCoupling](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#waveCoupling)                                | Controls the magnitude of winds passed to coupled wave models.    |
| [WindWaveMultiplier](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#WindWaveMultiplier) | This multiplier is applied after the winds have been read in by ADCIRC and time interpolated (if necessary) but without ADCIRC’s ramp function. It modifies the derived 10 minute averaged wind velocity values.                                        |
| [timeBathyControl](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#timeBathyControl)                             | Fortran namelist used to specify parameters related to simulations with time-varying bathymetry.                                                                                                                                                     |
| [NDDT](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NDDT)                                         | Controls the use of time-varying bathymetry and defines the spatial extent and time reference of the data in the time-varying bathymetry input file. Values of NDDT determine how the time-varying bathymetry file (fort.141) is interpreted.       |
| [BTIMINC](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#BTIMINC)                                      | Time increment (in seconds) between time-varying bathymetry datasets in the fort.141 file.                                                                                                                                                        |
| [BCHGTIMINC](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#BCHGTIMINC)                                   | Time increment (in seconds) over which bathymetry changes during a BTIMINC interval.                                                                                                                                                              |
| [tau0var](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#tau0var)                                      | tau0 value written out at every node in the fort.90 file.                                                                                                                                                                                          |
| [subdomainModeling](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#subdomainModeling)                            | Fortran namelist used to activate subdomain modeling, allowing the definition of a small subdomain nested within a larger domain.                                                                                                                 |
| [subdomainOn](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#subdomainOn)                                  | Logical variable that activates subdomain modeling when set to 'true'.                                                                                                                                                                              |
| [wetDryControl](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#wetDryControl)                                | Fortran namelist used to control the output and use of wet/dry state information in ADCIRC.                                                                                                                                                        |
| [inundationOutputControl](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#inundationOutputControl)                      | Fortran namelist used to activate the production of inundation-related output files and set the threshold for considering land as inundated.                                                                                                         |
| [TVWControl](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#TVWControl)                                   | Fortran namelist used to activate time-varying weirs and specify input/output files for weir behavior.                                                                                                                                             |
| [outputNodeCode](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#outputNodeCode)                               | Logical variable that activates the production of the nodecode.63 file for nodal wet/dry state.                                                                                                                                                   |
| [outputNOFF](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#outputNOFF)                                   | Logical variable that activates the production of the noff.100 file for elemental wet/dry state.                                                                                                                                                  |
| [noffActive](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#noffActive)                                   | Logical variable that can disable the elemental wet/dry state by setting NOFF to wet state always and everywhere.                                                                                                                                  |
| [inundationOutput](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#inundationOutput)                             | Logical variable that activates the production of inundation output files.                                                                                                                                                                         |
| [inunThresh](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#inunThresh)                                   | Numerical value that marks the threshold at which normally dry land is considered inundated.                                                                                                                                                       |
| [use_TVW](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#use_TVW)                                      | Logical variable that activates the time-varying weirs capability.                                                                                                                                                                                 |
| [TVW_file](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#TVW_file)                                     | Specifies the name of the input file that defines the behavior of time-varying weirs.                                                                                                                                                             |
| [nout_TVW](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#nout_TVW)                                     | Integer that activates the production of the fort.77 output file and specifies its format.                                                                                                                                                        |
| [touts_TVW](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#touts_TVW)                                    | Time in days since cold start after which output to the fort.77 file will start.                                                                                                                                                                  |
| [toutf_TVW](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#toutf_TVW)                                    | Time in days since cold start when output to the fort.77 file will end.                                                                                                                                                                           |
| [nspool_TVW](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#nspool_TVW)                                   | Time step increment at which output will be written to the fort.77 file.                                                                                                                                                                          |

Please note that the links provided will take you to the ADCIRC documentation for each respective parameter.


## Sections needed on clarifyin how to use parametrers:

- Tau parameter setting -> How to set appropriately?
- Ramping (DRAMP) parameters -> Differences and when to use which one?
- NOLIFA cases - Finite amplitude terms and wetting and drying of elements.
- NOLIBF cases  - Bottom stress parametrization cases.



## Solver options

Apologies for the confusion. Here's the updated markdown table with the first column as formatted links and the descriptions summarized:

| Parameter                                   | Description                                          |
|---------------------------------------------|------------------------------------------------------|
| [WindDragLimit](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#WindDragLimit)           | Controls the ceiling on the wind drag coefficient.    |
| [DragLawString](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#DragLawString)           | Controls the formulation for calculating nodal wind drag coefficients.    |
| [rhoAir](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#rhoAir)                                      | Sets the density of air.                             |
| [waveCoupling](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#waveCoupling)                                | Controls the magnitude of winds passed to coupled wave models.    |
| [WindWaveMultiplier](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#WindWaveMultiplier) | This multiplier is applied after the winds have been read in by ADCIRC and time interpolated (if necessary) but without ADCIRC’s ramp function. It modifies the derived 10 minute averaged wind velocity values.                                        |
| [timeBathyControl](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#timeBathyControl)                             | Fortran namelist used to specify parameters related to simulations with time-varying bathymetry.                                                                                                                                                     |
| [NDDT](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#NDDT)                                         | Controls the use of time-varying bathymetry and defines the spatial extent and time reference of the data in the time-varying bathymetry input file. Values of NDDT determine how the time-varying bathymetry file (fort.141) is interpreted.       |
| [BTIMINC](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#BTIMINC)                                      | Time increment (in seconds) between time-varying bathymetry datasets in the fort.141 file.                                                                                                                                                        |
| [BCHGTIMINC](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#BCHGTIMINC)                                   | Time increment (in seconds) over which bathymetry changes during a BTIMINC interval.                                                                                                                                                              |
| [tau0var](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#tau0var)                                      | tau0 value written out at every node in the fort.90 file.                                                                                                                                                                                          |
| [subdomainModeling](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#subdomainModeling)                            | Fortran namelist used to activate subdomain modeling, allowing the definition of a small subdomain nested within a larger domain.                                                                                                                 |
| [subdomainOn](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#subdomainOn)                                  | Logical variable that activates subdomain modeling when set to 'true'.                                                                                                                                                                              |
| [wetDryControl](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#wetDryControl)                                | Fortran namelist used to control the output and use of wet/dry state information in ADCIRC.                                                                                                                                                        |
| [inundationOutputControl](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#inundationOutputControl)                      | Fortran namelist used to activate the production of inundation-related output files and set the threshold for considering land as inundated.                                                                                                         |
| [TVWControl](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#TVWControl)                                   | Fortran namelist used to activate time-varying weirs and specify input/output files for weir behavior.                                                                                                                                             |
| [outputNodeCode](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#outputNodeCode)                               | Logical variable that activates the production of the nodecode.63 file for nodal wet/dry state.                                                                                                                                                   |
| [outputNOFF](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#outputNOFF)                                   | Logical variable that activates the production of the noff.100 file for elemental wet/dry state.                                                                                                                                                  |
| [noffActive](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#noffActive)                                   | Logical variable that can disable the elemental wet/dry state by setting NOFF to wet state always and everywhere.                                                                                                                                  |
| [inundationOutput](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#inundationOutput)                             | Logical variable that activates the production of inundation output files.                                                                                                                                                                         |
| [inunThresh](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#inunThresh)                                   | Numerical value that marks the threshold at which normally dry land is considered inundated.                                                                                                                                                       |
| [use_TVW](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#use_TVW)                                      | Logical variable that activates the time-varying weirs capability.                                                                                                                                                                                 |
| [TVW_file](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#TVW_file)                                     | Specifies the name of the input file that defines the behavior of time-varying weirs.                                                                                                                                                             |
| [nout_TVW](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#nout_TVW)                                     | Integer that activates the production of the fort.77 output file and specifies its format.                                                                                                                                                        |
| [touts_TVW](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#touts_TVW)                                    | Time in days since cold start after which output to the fort.77 file will start.                                                                                                                                                                  |
| [toutf_TVW](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#toutf_TVW)                                    | Time in days since cold start when output to the fort.77 file will end.                                                                                                                                                                           |
| [nspool_TVW](https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions#nspool_TVW)                                   | Time step increment at which output will be written to the fort.77 file.                                                                                                                                                                          |