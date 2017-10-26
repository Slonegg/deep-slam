#!/bin/bash

set -e
DIR="$(cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"
REV="$(git rev-parse HEAD)"

echo "OSTYPE is $OSTYPE"
if [[ "$OSTYPE" == "linux-gnu" ]]; then
    export BUILD_DIR_DEBUG="$DIR/_build/gcc_x64d"
    export BUILD_DIR="$DIR/_build/gcc_x64"
    INSTALL_DIR_DEBUG="$DIR/_release/deep_slam_gcc_x64d_$REV"
    INSTALL_DIR="$DIR/_release/deep_slam_gcc_x64_$REV"

    # build debug
    mkdir -p "$BUILD_DIR_DEBUG"
    cd "$BUILD_DIR_DEBUG"
    echo "======================= CONFIGURING ======================="
    cmake ../../ -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR_DEBUG" -DCMAKE_BUILD_TYPE=Debug $CMAKE_DEFINES
    echo "======================= BUILDING ======================="
    cmake --build . -- -j$(nproc)
    cd ../..

    # build release
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    echo "======================= CONFIGURING ======================="
    cmake ../../ -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" -DCMAKE_BUILD_TYPE=Release $CMAKE_DEFINES
    echo "======================= BUILDING ======================="
    cmake --build . -- -j$(nproc)
    cd ../..
else
    echo "Unknown operating system"
fi

# run tests if asked
if [[ "$*" == *"--test"* ]]
then
    echo "======================= TESTING ======================="
    cd "$BUILD_DIR_DEBUG"
    ctest -C Debug
    cd ../../
    cd "$BUILD_DIR"
    ctest -C Release
    cd ../../
fi

# create package, maybe installer in future
ARCHIVE_DEBUG="$INSTALL_DIR_DEBUG.tar.bz2"
ARCHIVE="$INSTALL_DIR.tar.bz2"
if [[ "$*" == *"--package"* ]] || [[ "$*" == *"--install"* ]]
then
    # install shrec-services into .release folder, clean it up before hand
    echo "======================= INSTALLING ======================="
    rm -rf "$DIR/_release"

    cd "$BUILD_DIR"
    cmake --build . --config "Release" --target install
    cd ../../

    if [[ "$BUILD_DIR" != "$BUILD_DIR_DEBUG" ]]
    then
        cd "$BUILD_DIR_DEBUG"
        cmake --build . --config "Debug" --target install
        cd ../../
    fi
fi
