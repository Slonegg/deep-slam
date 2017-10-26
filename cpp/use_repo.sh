#!/bin/bash

# download data
download_cmd=$(realpath ../libs/eaglecv/sysadmin/download_if_different.sh)

cd bin/foot_segments
"$download_cmd" https://s3-us-west-2.amazonaws.com/merlin-ext/data/mfs/swap_backgrounds.zip c6648593540d0065dcbb62892cc3e6fe swap_backgrounds.zip
cd ../..

# set binary dir variable
REV="$(git rev-parse HEAD)"
if [ -d ".build/gcc_x64" ]; then
    export BUILD_DIR="$(realpath .build/gcc_x64)"
fi
if [ -d ".build/gcc_x64d" ]; then
    export BUILD_DIR_DEBUG="$(realpath .build/gcc_x64d)"
fi
if [ -d ".build/v140_x64" ]; then
    export BUILD_DIR="$(realpath .build/v140_x64)"
fi
if [ -d ".build/v140_x64d" ]; then
    export BUILD_DIR_DEBUG="$(realpath .build/v140_x64d)"
fi

echo "BUILD_DIR env variable is set to " $BUILD_DIR
echo "BUILD_DIR_DEBUG env variable is set to " $BUILD_DIR_DEBUG
