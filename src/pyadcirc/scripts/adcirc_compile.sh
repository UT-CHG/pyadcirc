#!/usr/bin/bash

helpFunction()
{
	echo ""
	echo "Usage: $0 name [git] [tag]"
	echo -e "\tname - of adcirc to version/tag to build. Must be valid github version/tag at https://github.com/adcirc/adcirc-cg, or at the git url provided"
	echo -e "\tgit - URL. Default at https://github.com/adcirc/adcirc-cg)"
	echo -e "\ttag - 1 if is tagged version, 0 if not. Default is 0."
	exit 1 # Exit script after printing help
}

log () {
  echo "$(date) | ADCIRC_COMPILE | ${1} | ${2}"
}

# Print helpFunction in case parameters are empty
name=$1
if [ -z "$name" ]
then
	echo "Must specify version/tag name.";
	helpFunction
fi

gitURL=$2
if [ -z "$gitURL" ]
then
	gitURL="https://github.com/adcirc/adcirc-cg"
fi

tag=$3

# Make ExeDir if it doesn't exist
targetDir=$WORK/adcirc_execs
mkdir -p $targetDir
cd $targetDir

# Make directory for particular version
mkdir -p bin/$name

# Checkout adcirc repository in work directory if does not exist
if [ -d "repo" ]
then
	rm -rf repo
fi
log INFO "Cloning ADCIRC repo at $gitURL"
git clone $gitURL repo > ../adcirc_build.log 2>&1

# Move into git repo
cd repo

# Fetch all remote tags/branches
git config pull.rebase true
log INFO "Updating repo"
git fetch --all --tags >> adcirc_build.log 2>&1
git pull >> adcirc_build.log 2>&1

# Checkout version branch or tagged version
if [[ "$tag" -eq "1" ]]
then
	log INFO "Checking out tag $name"
	git checkout tags/$name -B  $name-branch >> adcirc_build.log 2>&1
else
	log INFO "Checking out branch $name"
	git checkout remotes/origin/$name -B  $name-build >> adcirc_build.log 2>&1
fi
if [ $? -eq 0 ]; then
    log INFO "Checked out $name successfully"
else
    err_msg=$(tail adcirc_build.log)
    log ERROR "Error checking out $name: $err_msg"
    exit 1
fi

# Make new cmake build dir
mkdir -p build
cd build

# Load the TACC modules needed for compilation
module load netcdf cmake

# Configure cmake - Options for TACC Compilation (knl stampede2)
cmake .. -DCMAKE_C_COMPILER=icc \
	 -DCMAKE_CXX_COMPILER=icpc \
	 -DCMAKE_Fortran_COMPILER=ifort \
	 -DENABLE_GRIB2=OFF \
	 -DENABLE_DATETIME=ON \
	 -DENABLE_OUTPUT_NETCDF=ON \
	 -DNETCDFHOME=$TACC_NETCDF_DIR \
	 -DBUILD_ADCPREP=ON \
	 -DBUILD_PADCIRC=ON \
	 -DBUILD_PADCSWAN=ON \
	 -DCMAKE_Fortran_FLAGS_RELEASE="$TACC_VEC_FLAGS -O3" \
	 -DCMAKE_C_FLAGS_RELEASE="$TACC_VEC_FLAGS -O3 -DNDEBUG" >> ../adcirc_build.log 2>&1
if [ $? -eq 0 ]; then
    log INFO "cmake configuration complete"
else
    err_msg=$(tail ../adcirc_build.log)
    log ERROR "make error, check adcirc_build.log: $err_msg"
    log ERROR "cmake configuration error, check adcirc_build.log"
    exit 1
fi


# Build the code using 6 cores and move executables when done. Do in background
# Let any errors print out to stderr so we can pick them up back in execution of script
make -j6 >> ../adcirc_build.log 2>&1
if [ $? -eq 0 ]; then
    log INFO "make complete"
else
    err_msg=$(tail ../adcirc_build.log)
    log ERROR "make error, check adcirc_build.log: $err_msg"
    exit 1
fi

log INFO "Copying executables to dest dir ${WORK}/../adcirc/execs/${TACC_SYSTEM}-${name}/"
mkdir -p "${WORK}/../adcirc/execs/${TACC_SYSTEM}-${name}/"
cp adcprep padcirc padcswan "${WORK}/../adcirc/execs/${TACC_SYSTEM}-${name}/"
