# 1. Seal Remove (Yolo + Unet) (tensaorrt)
[Seal Remove]以先检测（Yolo）再去除（Unet）的思路，实现快速且针对大尺寸图像的印章去除。<br>

# 2. Requirements

TensorRT 7.2<br>
Cuda 11.1<br>
Python 3.7<br>
opencv 3.4<br>
cmake 3.18<br>

# 3. Preparation
请先阅读README，并生成Yolo与Unet各自的engine，int8则需要对应的table校准文件。<br>

# 4. Inference
创建build文件夹，然后再build文件夹下进行遍历，遍历之后会在本文件夹下生成可执行文件。<br>
```
mkdir build
cd build
cmake ..
make
```

执行需要指定两个engine与图像文件夹。<br>
```
./seal_remove  ../yolov5m_fp16.engine ../unet_fp16.engine ../data
```

# 5. Efficiency
下面列出了加速后的时间损耗 (测试环境为：Tesla V100)

  | FP32 | FP16 | INT8
  | ----- | ------  | ------
  | 512x512 | 512x512 | 512x512
  | 24ms  | 12ms  | 14ms
