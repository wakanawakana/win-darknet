set Path = %Path%;%~dp0build\Release\x64;
build\Release\x64\darknet_gpu.exe -i 0 detector test cfg\coco.data cfg\yolo.cfg yolo.weights data\dog.jpg
