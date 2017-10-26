## Preparing repository
To create python environment
```
cd python
source ./use_repo.sh
```

Do this before executing any script to activate virtual environment.

## Testing

Simply run `py.test test`

## Creating Kinect RGB-D Dataset

1. Build libfreenect:
    ```
    git clone git@github.com:OpenKinect/libfreenect.git
    cd libfreenect && mkdir _build && cd _build
    cmake ..
    make -j8
    ```

2. Use provided `fakenect-record` example to record RGB-D data:
    ```
    ./bin/fakenect-record rgbd_data
    ```

3. ...