#!/usr/bin/bash


# Where to store executables built
ADCIRC_EXECS_DIR=adcirc_execs

# Version to build - Should correspond to a git branch or tag
VERSION=v55_nodal_beta

cd $WORK

# Make Execs Dir if it doesn't exist
mkdir -p $ADCIRC_EXECS_DIR
cd $ADCIRC_EXECS_DIR

# Make directory for particular version
mkdir -p bin/$VERSION

# Checkout adcirc repository in work directory if does not exist
if [ ! -d "repo" ]
then
	log INFO "Cloning ADCIRC Repo"
	git clone -q https://github.com/cdelcastillo21/adcirc-cg repo 
#	git clone -q https://github.com/adcirc/adcirc-cg repo >> adcirc_build.log
fi

# Move into build directory 
cd repo 

# Fetch all remote tags/branches
git fetch --all --tags > adcirc_build.log 2>&1

# Checkout branch, for checking out tagged version (For stable releases), see commented out line below
# git checkout remotes/origin/$VERSION -b  $VERSION-branch > adcirc_build.log 2>&1
git pull >> adcirc_build.log 2>&1
git checkout tags/v55.00 -b  v55.00-branch > adcirc_build.log 2>&1

# Make new cmake build dir
mkdir -p build
cd build

# Load the TACC modules needed for compilation
module load netcdf cmake

# Configure cmake - Options for TACC Compilation (knl stampede2)
cmake .. -DCMAKE_C_COMPILER=icc \
	-DCMAKE_CXX_COMPILER=icpc \
	-DCMAKE_Fortran_COMPILER=ifort \
	-DENABLE_GRIB2=ON \
	-DENABLE_DATETIME=ON \
	-DENABLE_OUTPUT_NETCDF=ON \
	-DNETCDFHOME=$TACC_NETCDF_DIR \
	-DBUILD_ADCPREP=ON \
	-DBUILD_PADCIRC=ON #- DBUILD_PADCSWAN=ON \
	-DCMAKE_Fortran_FLAGS_RELEASE="$TACC_VEC_FLAGS -O3" \
	-DCMAKE_C_FLAGS_RELEASE="$TACC_VEC_FLAGS -O3 -DNDEBUG" > ../adcirc_build.log 2>&1


# Build the code using 6 cores and move executables when done. Do in background
# Let any errors print out to stderr so we can pick them up back in execution of script
make -j6 >> ../adcirc_build.log 2>&1 && cp adcprep padcirc padcswan ../../bin/$VERSION/
