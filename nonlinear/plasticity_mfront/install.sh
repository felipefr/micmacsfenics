#! /usr/bin/env bash

# This scripts is meant to install the development versions of TFEL
# and MGIS.
# The user must change the PREFIX variable to match its needs.
# Once the script is finished you can use TFEL and MGIS by running:
# source <PREFIX>/codes/mgis/master/install/env.sh

# Before running the script, the user must install the appropriate
# prerequisites.
# On ubuntu, please run:
# sudo apt-get install cmake libboost-all-dev g++ gfortran
# sudo apt-get install git libqt5svg5-dev qtwebengine5-dev
# sudo apt-get install python3-matplotlib

# Note: by default, the FEniCS interface is disabled.
# If FEniCS is installed, you can unable it by changing
# `-Denable-fenics-bindings=OFF` to
# `-Denable-fenics-bindings=ON`
#

# Those bindings are *not* required to run the examples of Jérémy
# Bleyer which are based on the `python` bindings.
# See https://thelfer.github.io/mgis/web/FEniCSBindings.html
# for details.

set -e
PREFIX=$HOME

PYTHON_VERSION=$(python3 --version|awk '{print $2}'|awk 'BEGIN{FS="."} {print $1 "." $2}')

export TFELSRC=$PREFIX/codes/tfel/master/src
export MGISSRC=$PREFIX/codes/mgis/master/src
export TFELBASEHOME=$PREFIX/codes/tfel/master
export MGISBASEHOME=$PREFIX/codes/mgis/master

mkdir -p $TFELSRC
mkdir -p $MGISSRC

pushd $TFELSRC
if [ -d tfel ]
then
	pushd tfel
	git pull
	popd 
else
	git clone https://github.com/thelfer/tfel.git
fi
popd

pushd $MGISSRC
if [ -d  MFrontGenericInterfaceSupport ]
then
	pushd MFrontGenericInterfaceSupport
	git pull
	popd 
else
	git clone https://github.com/thelfer/MFrontGenericInterfaceSupport.git
fi
popd

tfel_rev=$(cd $TFELSRC/tfel;git rev-parse HEAD)
mgis_rev=$(cd $MGISSRC/MFrontGenericInterfaceSupport;git rev-parse HEAD)

export TFELHOME=$TFELBASEHOME/install-${tfel_rev}
export MGISHOME=$MGISBASEHOME/install-${mgis_rev}

pushd $TFELBASEHOME
tfel_previous=$(ls -1dtr install-* | tail -1)
tfel_previous_rev=${tfel_previous#install-}
popd

echo "tfel current  revision : $tfel_rev" 
echo "tfel previous revision : $tfel_previous_rev" 
if [ x"$tfel_previous_rev" != x"$tfel_rev" ];
then
  pushd $TFELSRC
  mkdir -p build
  pushd build
  cmake ../tfel -DCMAKE_BUILD_TYPE=Release -Dlocal-castem-header=ON -Denable-aster=ON -Denable-abaqus=ON -Denable-calculix=ON -Denable-ansys=ON -Denable-europlexus=ON -Denable-python=ON -Denable-python-bindings=ON -DPython_ADDITIONAL_VERSIONS=$PYTHON_VERSION -DCMAKE_INSTALL_PREFIX=$TFELHOME
  make
  make install
  popd
  popd
  
  cat >> ${TFELHOME}/env.sh <<EOF
  export TFELHOME=${TFELHOME}
  export PATH=${TFELHOME}/bin:\$PATH
  export LD_LIBRARY_PATH=${TFELHOME}/lib:\$LD_LIBRARY_PATH
  export PYTHONPATH=${TFELHOME}/lib/python${PYTHON_VERSION}/site-packages/:\$PYTHONPATH
EOF
  
  pushd $TFELBASEHOME
  count=$(ls -1d install-* | wc -l)
  if (( $count > 4 )) ; then 
    find . -maxdepth 1 -ctime +60 -name "install-*" -exec rm -rf {} \;
  fi
  last=$(ls -1dtr install-* | tail -1)
  ln -nsf "${last}" install
fi

source ${TFELHOME}/env.sh

pushd $MGISBASEHOME
mgis_previous=$(ls -1dtr install-* | tail -1)
mgis_previous_rev=${mgis_previous#install-}
popd

if [ x"$tfel_previous_rev" != x"$tfel_rev" ] || [ x"$mgis_previous_rev" != x"$mgis_rev" ];
then
  pushd $MGISSRC
  mkdir -p build
  pushd build
  cmake ../MFrontGenericInterfaceSupport -DCMAKE_BUILD_TYPE=Release -Denable-python-bindings=ON -Denable-fortran-bindings=ON -Denable-c-bindings=ON -Denable-fenics-bindings=OFF -DCMAKE_INSTALL_PREFIX=$MGISHOME
  make
  make install
  popd
  popd
  
  cat >> ${MGISHOME}/env.sh <<EOF
  export MGISHOME=${MGISHOME}
  source ${TFELHOME}/env.sh
  export PATH=${MGISHOME}/bin:\$PATH
  export LD_LIBRARY_PATH=${MGISHOME}/lib:\$LD_LIBRARY_PATH
  export PYTHONPATH=${MGISHOME}/lib/python${PYTHON_VERSION}/site-packages/:\$PYTHONPATH
EOF
  
  pushd $MGISBASEHOME
  count=$(ls -1d install-* | wc -l)
  if (( $count > 4 )) ; then 
    find . -maxdepth 1 -ctime +60 -name "install-*" -exec rm -rf {} \;
  fi
  last=$(ls -1dtr install-* | tail -1)
  ln -nsf "${last}" install
fi

