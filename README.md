# LFRendering_CuPy
GPU-based Light Field Rendering with CuPy, pyTOFlow



# Prerequisites



## NumPy, Cupy
To store and calculate images, NumPy and CuPy modules are used. CuPy is used for calculation in GPU environment.

## PyTorch
Our implementation is based on PyTorch 0.4.1, but we can use the other versions. The current environment is PyTorch 1.7.1.
(python 3.6.12)

## TOFlow
The part of interpolation is based on pyTOFlow, so we need modules and files in https://github.com/Coldog2333/pytoflow.



# Usage

After 
After downloading the https://github.com/Coldog2333/pytoflow, insert LFRendering_TOFlow_0326.py and M3DLF_S3(LightField photo folder) and run it.
```
python LFRendering_TOFlow_0326.py --o path_out.jpg --gpuID 0
```

## Options
Because there were modifications in pyTOFlow, there are only two options.

+ --o [optional]: filename of the predicted frame. default: out.png, saving in the same directory of the input frames.
+ --gpuID [optional]: No of the GPU you want to use. default: gpuID = 0.

# cf)

![image](https://user-images.githubusercontent.com/81347659/114809778-c863fe00-9de5-11eb-97b5-0e9be0ffd0fe.png)



