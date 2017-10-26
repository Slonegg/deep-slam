# deep-slam
Attempt to use deep learning for SLAM

## Prerequisites on Ubuntu
1. Install CUDA 8.0:
    - Download binaries from [CUDA download site](https://developer.nvidia.com/cuda-downloads)
    - Run:
    ```
    sudo apt-get purge nvidia*
    sudo dpkg -i <cuda deb>
    sudo apt-get update
    sudo apt-get install cuda
    ```
    - Add following environment variables definitions to `/etc/environment`
    ```
    CUDA_HOME=/usr/local/cuda-8.0
    LD_LIBRARY_PATH=${CUDA_HOME}/lib64
    PATH=${CUDA_HOME}/bin:${PATH}
    ```
2. Install cuDNN 5.1:
    - Download cuDNN 5.1 [here](https://developer.nvidia.com/cudnn)
    - Extract archive and navigate to extracted folder:
    ```
    sudo cp -P include/cudnn.h /usr/include
    sudo cp -P lib64/libcudnn* /usr/lib/x86_64-linux-gnu/
    sudo chmod a+r /usr/lib/x86_64-linux-gnu/libcudnn*
    ```
    
## See also

- [Python Readme](python/README.md)
- [C++ Readme](cpp/README.md)
