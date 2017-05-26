# Domain Transfer Network (DTN) 
TensorFlow implementation of [Unsupervised Cross-Domain Image Generation.](https://arxiv.org/abs/1611.02200)
![alt text](jpg/dtn.jpg)

## Requirements
* [Python 2.7](https://www.continuum.io/downloads)
* [TensorFlow 0.12](https://github.com/tensorflow/tensorflow/tree/r0.12)
* [Pickle](https://docs.python.org/2/library/pickle.html)
* [SciPy](http://www.scipy.org/install.html)

<br>

## Usage

#### Clone the repository
```bash
$ git clone https://github.com/yunjey/dtn-tensorflow.git
$ cd dtn-tensorflow
```

#### Download the dataset
```bash
$ chmod +x download.sh
$ ./download.sh
```

#### Resize MNIST dataset to 32x32 
```bash
$ python prepro.py
```

#### Pretrain the model f
```bash
$ python main.py --mode='pretrain'
```

#### Train the model G and D
```bash
$ python main.py --mode='train'
```

#### Transfer SVHN to MNIST
```bash
$ python main.py --mode='eval'
```
<br>

## Results

#### From SVHN to MNIST 

![alt text](jpg/svhn_mnist_2900.gif)

![alt text](jpg/svhn_mnist_2900.png)

![alt text](jpg/svhn_mnist_3700.png)

![alt text](jpg/svhn_mnist_5300.png)


#### From Photos to Emoji (in paper)

![alt text](jpg/emoji_1.png)

![alt text](jpg/emoji_2.png)





git : https://github.com/ITERRYI/domain-transfer-network.git
폴더위치 : /home/ubuntu/_src/domain-transfer-network
환경 : Python2.7       TensorFlow 0.12

[Clone작업시작]<실행> git clone https://github.com/ITERRYI/domain-transfer-network.git폴더위치 : /home/ubuntu/_src/domain-transfer-network

[dataset 다운로드]<실행> chmod +x download.sh       ./download.sh

[리사이즈 MNIST dataset 32X32]<실행> python prepro.py
<오류 발생>Traceback (most recent call last):  File "prepro.py", line 3, in <module>    from PIL import ImageImportError: No module named PIL
<설치> pip install Pillow

[Pretrain the model f]<실행> python main.py --mode='pretrain'
<오류발생>Traceback (most recent call last):  File "main.py", line 3, in <module>    from solver import Solver  File "/home/ubuntu/_src/domain-transfer-network/solver.py", line 6, in <module>    import scipy.io
<설치>  pip install scipy
<실행> python main.py --mode='pretrain' >> main_mode_pretrain.txt
returning NUMA node zeroI tensorflow/core/common_runtime/gpu/gpu_device.cc:885] Found device 0 with properties:name: Tesla K80major: 3 minor: 7 memoryClockRate (GHz) 0.8235pciBusID 0000:00:1e.0Total memory: 11.17GiBFree memory: 11.11GiBI tensorflow/core/common_runtime/gpu/gpu_device.cc:906] DMA: 0I tensorflow/core/common_runtime/gpu/gpu_device.cc:916] 0:   YI tensorflow/core/common_runtime/gpu/gpu_device.cc:975] Creating TensorFlow device(/gpu:0) -> (device: 0, name: Tesla K80, pci bus id: 0000:00:1e.0)
<샐행결과> main_mode_pretrain.txt 확인

[Train the model G and D]<실행> python main.py --mode='train'  >> main_mode_train.txt
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcublas.so locallyI tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcudnn.so locallyI tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcufft.so locallyI tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcuda.so.1 locallyI tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcurand.so locallyI tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zeroI tensorflow/core/common_runtime/gpu/gpu_device.cc:885] Found device 0 with properties:name: Tesla K80major: 3 minor: 7 memoryClockRate (GHz) 0.8235pciBusID 0000:00:1e.0Total memory: 11.17GiBFree memory: 11.11GiBI tensorflow/core/common_runtime/gpu/gpu_device.cc:906] DMA: 0I tensorflow/core/common_runtime/gpu/gpu_device.cc:916] 0:   YI tensorflow/core/common_runtime/gpu/gpu_device.cc:975] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla K80, pci bus id: 0000:00:1e.0)


[Transfer SVHN to MNIST]<실행> python main.py --mode='eval'  >> main_mode_eval.txt
