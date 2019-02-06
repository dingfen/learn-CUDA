CUDA

### 基本CUDA入门

本节展示了最基本的CUDA入门知识。

### 环境要求

CUDA 8.0

## 

## CUDA基本概念

- GPU并行化
- CUDA C/C++的函数修饰符

欢迎来到CUDA的世界。

随着显卡的发展，GPU越来越强大，而且GPU为显示图像做了优化。在计算上已经超越了通用的CPU。如此强大的芯片如果只是作为显卡就太浪费了，因此NVIDIA推出CUDA，让显卡可以用于图像计算以外的目的。

**我们用host指代CPU及其内存，而用device指代GPU及其内存**。CUDA程序中既包含host程序，又包含device程序，它们分别在CPU和GPU上运行。同时，host与device之间可以进行通信，这样它们之间可以进行数据拷贝。

GPU并行化的工作流程：

1. CPU发送一种称为`kernel`的函数到GPU。
2. GPU同时运行该函数的多个版本，称为`threads`。`thread`可以组合成`block`，一个`kernel`里的所有`thread`称为一个`grid`。

`__global__` 是CUDA C/C++的函数修饰符，表示该函数为一个`kernel`函数，且

- 该函数会在GPU(`device`)上执行。
- 必须返回void。
- 由主机(`host`)代码调用。

在调用`kernel`函数时，函数名后的`<<<b, t>>>`：

- b代表`block`的数目。
- t代表每个`block`中`thread`的数目。

### [程序要求](./cuda.c)

`kernel`函数需要运行在4个`block`上，每个`block`有2个`thread`。

### 输出结果

```
Hello, World!
```

## 函数修饰符

- `__host__`修饰符
- `__global__`修饰符
- `__device__`修饰符

上节简单介绍了`__global__`函数修饰符。下面详细介绍函数修饰符。

```CUDA
__host__
```

1. 运行在CPU上，每次调用运行一次。
2. 只能被CPU调用。
3. 所有未显式标明函数前置修饰符的函数均为host函数。

```CUDA
__global__
```

1. 用于声明一个`kernel`函数。
2. 运行在GPU上，每次调用可以运行多次（由`<<<#, #>>>`决定）。
3. 只能被CPU调用。

```CUDA
__device__
```

1. 运行在GPU上，每次调用运行一次。
2. 只能被GPU调用。

### [程序要求](./func.c)

在GPU上运行`dev1`与`dev2`函数10次。

### 输出结果

```
Hello, World!
```

## 内存分配函数

- `cudaMalloc`分配设备上的内存
- `cudaMemcpy`将不同内存段的数据进行拷贝
- `cudaFree`释放先前在设备上申请的内存空间
- `cudaMallocPitch`分配设备上的内存
- `cudaMalloc3D` 分配设备上的内存
### 函数说明

```C
__host__ cudaError_t cudaMalloc (void **devPtr, size_t size)
```

* 该函数主要用来分配**设备上的内存（即显存中的内存）**。该函数被声明为了`__host__`，即表示被host所调用，即在CPU中执行的代码所调用。
* 返回值：为`cudaError_t`类型，实质为`cudaError`的枚举类型，其中定义了一系列的错误代码。如果函数调用成功，则返回`cudaSuccess`。
* 第一个参数，`void **` ，用于接受该函数所分配的内存地址。
* 第二个参数，`size_t`，用于指定分配内存的大小，单位为字节。

```C
__host__ cudaError_t cudaMemcpy (void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
```

* 该函数**主要用于将不同内存段的数据进行拷贝，内存可用是设备内存，也可用是主机内存**
* 第一个参数，void*类型，dst：为目的内存地址
* 第二个参数，const void *类型，src：源内存地址
* 第三个参数，size_t类型，count：将要进行拷贝的字节大小
* 第四个参数，`enum cudaMemcpyKind`，拷贝的类型，决定拷贝的方向。
    `cudaMemcpyKind`类型如下：`cudaMemcpyHostToHost`, `cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToHost`, `cudaMemcpyDeviceToDevice`, `cudaMemcpyDefault`。

```C
__host__ cudaError_t cudaFree (void* devPtr)
```

* 该函数用来**释放先前在设备上申请的内存空间**（通过cudaMalloc、cudaMallocPitch等函数），注意，不能释放通过标准库函数malloc进行申请的内存。

* 返回值：为错误代码的类型值。

* 第一个参数，`void**`，指向需要释放的设备内存地址。

```C
cudaError_t  cudaMallocPitch(void **devPtr, size_t *pitch, size_t width size_t height);
```

* 该函数**用来分配指定大小的线性内存，宽度至少为width，高度为height。**
* 在分配内存时会适当的填充一些字节来保证对其要求，从而在按行访问时，或者在二维数组和设备存储器的其他区域间复制是，保证最佳的性能
* 实际的分配的内存大小为：sizeof(T) \* pitch \* height，则访问2D数组中任意一个元素[Row,Column]的计算公式如下：
    ​    <center>T* pElement = (T*)((char*)BaseAddress + Row * pitch) + Column。</center>
* 第一个参数，`void**` 用来接受被分配内存的其实地址
* 第二个参数，`size_t*` 用来接受实际行间距，即被填充后的实际宽度（单位字节），大于等于第三个参数width
* 第三个参数，`size_t` 请求分配内存的宽度（单位字节），如2D数组的列数
* 第四个参数，`size_t` height：请求分配内存的高度（单位字节），如2D数组的行数


```C
cudaError_t cudaMalloc3D(struct cudaPitchedPtr* pitchedDevPtr, struct cudaExtent extent);  
```

* 该函数用来申请设备上的1D、2D、3D内存对象，同cudaMallocPitch函数一样，为了最佳的性能，会填充一些字节。
* 第一个参数，`cudaPitchPtr*` pitchedDevPtr：**作为传出参数**，用于记录分配得到的设备内存信息，具体结构如下：
    ```C
    struct  cudaPitchedPtr  
    {  
        void   *ptr;      //指向分配得到的设备内存地址  
        size_t  pitch;    //实际被分配的宽度，单位字节  
        size_t  xsize;    //逻辑宽度，记录有效的宽度，单位字节  
        size_t  ysize;    //逻辑高度，记录有效的高度，单位高度  
    };  
    ```

* 第二个参数，`cudaExtent` extent：**作为传入参数**，传入所请求申请的内存信息，包括width、height、depth；具体结构如下：
    ```C
    struct cudaExtent  
    {  
        size_t width;     //请求的宽度，单位字节  
        size_t height;    //请求的高度，单位字节  
        size_t depth;     //请求的深度，单位字节  
    };
    ```


### [程序示例1](./memalloc.c)

```
a = 2
```
### [程序示例2](./memalloc2.cpp)

```
width: 40
height: 40
pitch: 512
```

### [程序示例3](./memalloc3.cpp)
```
width: 40
height: 88
depth: 132

pitch: 512
xsize: 40
ysize: 88
```

## 内置变量

- `blockIdx`
- `threadIdx`
- `gridDim`
- `blockDim`

![img](https://www.easyhpc.org/static/upload/md_images/20180526112210.png)

`maxThreadsPerBlock`:每个`block`上的`thread`最大值，默认为1024。

对于一个`kernel`函数，具有B个`block`，每个`block`有T个`thread`：

`blockIdx.x`：当前`block`在x方向上的ID，取值为0~B-1。

`threadIdx.x`：当前`block`上的当前`thread`在x方向上的ID，取值为0~T-1。

`gridDim.x`：当前`grid`的`block`在x方向上的数量。

`blockDim.x`：当前`block`的`thread`在x方向上的数量。

---

### 函数说明

`__syncthreads()`是 CUDA 的内置命令。

block内部用于线程同步。

同一block内所有线程执行至`__syncthreads()`处等待全部线程执行完毕后再继续。

### [程序示例1](./var.cpp)

```
a + b = c
    0 +     0 =     0
   -1 +     1 =     0
   -2 +     4 =     2
   -3 +     9 =     6
   -4 +    16 =    12
   -5 +    25 =    20
   -6 +    36 =    30
   -7 +    49 =    42
   -8 +    64 =    56
   -9 +    81 =    72
```

### [程序示例2](./var2.cpp)
```
v =
0.840
0.394
0.783
0.798
0.912
0.198
0.335
0.768

Pairwise sum = 5.029
```

## 常量内存

- `__constant__`修饰符
- `cudaMemcpyToSymbol`常量内存初始化

常量内存驻留在GPU内存中，用`__constant__`来修饰定义，代表着定义了一个常量内存。对于常量内存，不需要在使用完成后用`cudafree`释放空间。

`kernel`只能从常量内存中读取数据，**因此其初始化必须在host端使用`cudaMemcpyToSymbol`调用**

### 函数说明

```C
cudaError_t cudaMemcpyToSymbol(const void* symbol, const void* src, size_t count);
```

* 该函数拷贝从指针src的count个字节到symbol指向的地址，symbol指向的变量是在device中的global或者constant Memory。

### [运行结果](./constant.cpp)

```
0
15
30
45
60
75
90
105
120
135
150
165
180
195
210
225
240
255
270
285
300
315
330
345
360
375
390
405
420
435
450
465
480
495
510
525
540
555
570
585
```

## 全局内存

- `cudaMemcpyToSymbol` 数据拷贝到全局内存中
- `cudaMemcpyToSymbol` 数据拷贝到常量内存中

GPU与CPU都可以对全局内存进行写操作。

同样使用`cudaMemcpyToSymbol`将数据拷贝到全局内存中。

申请的是GPU内存，`cudaMemcpyToSymbol`拷贝就是从host拷贝到全局内存。申请的是常量内存，`cudaMemcpyToSymbol`拷贝就是从host拷贝到常量内存。

### [程序示例](./global.cpp)

```
Host: copied 3.140000 to the global variable
Device: the value of the global variable is 3.140000
Host: the value changed by the kernel to 5.140000
```

## 共享内存

- `__shared__`变量声明说明符

共享内存实际上是可受用户控制的一级缓存。申请共享内存后，其内容在每一个用到的block被复制一遍，使得在每个block内，每一个thread都可以访问和操作这块内存，而无法访问其他block内的共享内存。这种机制就使得一个block之内的所有线程可以互相交流和合作。

在设备代码中声明共享内存要使用`__shared__`变量声明说明符。

共享内存有两种方法：静态与动态。

**如果共享内存数组的大小在编译时就可以确定，我们就可以显式地声明固定大小的数组。**

如果共享内存的大小在编译时不能确定，则需要用动态分配共享内存的方式。在这种情况下，**每个线程块中共享内存的大小必须在核函数第三个执行配置参数中指定(以字节为单位)。如下所示：**

```
dynamicReverse<<<1, n, n*sizeof(int)>>>(d_d, n);
```

*共享内存注意同步问题。*

### [程序示例](./shared.cpp)

```
static success
dynamic success
```