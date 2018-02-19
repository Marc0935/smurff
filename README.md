# SMURFF - Scalable Matrix Factorization Framework
[![Build Status](https://travis-ci.org/ExaScience/smurff.svg?branch=master)](https://travis-ci.org/ExaScience/smurff)

## Bayesian Factorization with Side Information

Highly optimized and parallelized methods for Bayesian Factorization, including 
[BPMF](https://www.cs.toronto.edu/~amnih/papers/bpmf.pdf), 
[Macau](https://arxiv.org/abs/1509.04610) and
[GFA](https://arxiv.org/pdf/1411.5799.pdf).
The package uses optimized OpenMP/C++ code with a Cython wrapper to factorize large scale matrices.

Macau method is able to perform **matrix** and **tensor** factorization while incorporating high-dimensional side information to the factorization.

# Examples
For examples see [documentation](http://macau.readthedocs.io/en/latest/source/examples.html).

# Installation

## Python wrapper installation on Ubuntu

1. Build C++ SMURFF library using [Source installation on Ubuntu using cmake](#source-installation-on-ubuntu-using-cmake)
2. Go to SMURFF build directory that contains CMake generated Makefile and run `sudo make install`. By invoking this command you install SMURFF static library and headers to `/usr/local/`
3. Go to [python/smurff](https://github.com/ExaScience/smurff/tree/master/python/smurff) directory
4. Run `python setup.py build install`
5. Go to [python/smurff/examples](https://github.com/ExaScience/smurff/tree/master/python/smurff/examples) directory and run `python smurff-mm.py` to verify that SMUFF Python module was installed correctly

## Source installation on Ubuntu using cmake

Before continuing installation please check that 
 - cmake version is at least 3.6
 - eigen3 version 3.3.3 or later is installed. 

This is required due to the fixed Find scripts for BLAS libraries that are present in latest version.

cmake has multiple switches:

Build type switches:
* CMAKE_BUILD_TYPE - Debug/Release

Algebra library switches (select only one):
* ENABLE_BLAS - ON/OFF (only blas is not enough because currently smurff depends on some lapack functions)
* ENABLE_LAPACK - ON/OFF (should include lapack and blas libraries when linking)
* ENABLE_OPENBLAS - ON/OFF (should include openblas library when linking. openblas also contains implementation of lapack called relapack)

```bash
# install dependencies:
sudo apt-get install libopenblas-dev autoconf gfortran

# checkout and install Smurff
git clone https://github.com/ExaScience/smurff.git
cd smurff/lib/smurff-cpp/
mkdir build
cd build
cmake ../cmake -DENABLE_OPENBLAS=ON -DCMAKE_BUILD_TYPE=Debug
make
make test

# test Smurff:
wget http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-346targets.mm
Debug/smurff --train chembl-IC50-346targets.mm
```

## Source installation on Windows using Visual Studio and cmake

Before continuing installation please check that 
 - cmake version is at least 3.6
 - Visual Studio version is 2013 or better 2015 because code uses some c++11 features

### Install boost

Download latest version of boost from http://www.boost.org/

Start Visual Studio command prompt and execute the following commands.
```
bootstrap.bat
b2 toolset=msvc-14.0 address-model=64 --build-type=complete stage
```
This will compile and install boost libraries into build directory.

If you have Visual Studio different from 2015 - select proper toolset.

Configure boost environment variables as in the example:
```
BOOST_INCLUDEDIR=E:\boost_1_65_0
BOOST_LIBRARYDIR=E:\boost_1_65_0\stage\lib
```

### Install eigen3

Execute the following commands from command prompt:
```
git clone https://github.com/RLovelett/eigen.git
cd eigen
mkdir build
cd build
cmake ../ -G "Visual Studio 14 2015 Win64"
```
If you have Visual Studio different from 2015 - select proper generator.

Build INSTALL target in Visual Studio in Release configuration.

This will build all projects and install them in Program Files by default.

Configure eigen3 environment variables as in the example:
```
EIGEN3_INCLUDE_DIR=C:\Program Files\Eigen3\include\eigen3
```

### Install MinGW-64

MinGW-64 is required to build OpenBLAS library. MinGW-64 is chosen because it is easy to install fortran compiler dependency.
Fortran compiler is requried for building ReLAPACK part of OpenBLAS. Other option (not described here) is to install fortran compiler directly. There are few binary distributions described here: http://fortranhelp.blogspot.ru/2010/09/i-have-just-installed-gfortran-on.html

Download installer at http://www.msys2.org/

Configure msys2 exactly as described in the guide.

Install corresponding packages with pacman

```
pacman -S gcc
pacman -S gcc-fortran
pacman -S make
pacman -S autoconf
pacman -S automake
```

Add path to MinGW-64 binaries to PATH variable as in the example:
```
C:\msys64\usr\bin
```

### Install OpenBLAS
Open MinGW-64 command prompt

Execute the following commands:

```
git clone https://github.com/xianyi/OpenBLAS.git
cd OpenBLAS
make
make PREFIX=/e/openblas_install_64 install
```

You can change installation prefix if you want.

Set environment variables as in the example:
```
BLAS_INCLUDES=E:\openblas_install_64\include
BLAS_LIBRARIES=E:\openblas_install_64\lib\libopenblas.dll.a
```

Add path to OpenBLAS binaries as in the example:
```
E:\openblas_install_64\bin
```

### Install Smurff

Execute the following commands from command prompt:

```
git clone https://github.com/ExaScience/smurff.git
cd smurff\lib\smurff-cpp\cmake
mkdir build
cd build
cmake ../ -G "Visual Studio 14 2015 Win64" -DENABLE_OPENBLAS=ON -DENABLE_VERBOSE_COMPILER_LOG=ON
```

If you have Visual Studio different from 2015 - select proper generator.

Build INSTALL target in Visual Studio in Release configuration.

This will build all projects and install them in Program Files by default.

## Installation using Conda
```bash
conda install -c vanderaa smurff 
```

# Contributors
- Jaak Simm (Macau C++ version, Cython wrapper, Macau MPI version, Tensor factorization)
- Tom Vander Aa (OpenMP optimized BPMF, Matrix Cofactorization and GFA, Code Reorg)
- Adam Arany (Probit noise model)
- Tom Haber (Original BPMF code)
- Andrei Gedich
- Ilya Pasechnikov

# Acknowledgements
This work was partly funded by the European projects ExCAPE (http://excape-h2020.eu) and
EXA2CT, and the Flemish Exaptation project.

