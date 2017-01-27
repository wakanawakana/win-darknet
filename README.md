*Darknet Implement Win32/64*

*Supports*
* yolo V2
* use OPENCV option
* use GPU option
* use CUDNN option

*Tested pattern*
* Win32 (VS2013）
* Win32 (OPENCV & VS2013）
* Win32 (GPU(Cuda6.5) & OPENCV & VS2013）
* x64 (VS2013）
* x64 (OPENCV & VS2013）
* x64 (GPU(Cuda6.5) & OPENCV & VS2013）
* x64 (GPU(Cuda8.0) & CUDNN & OPENCV & VS2013）

*What Modify*
* I made it possible to run with only CPU even in GPU build.
* Implementation of functions missing in Windows.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

"yolov2wgpu.bat" is a test batch of yolov 2.  
Download the weits file to the TOP hierarchy and use it.  
Surely predictions images will be generated.  
