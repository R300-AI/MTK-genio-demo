# Genio EVK Setup Guide
##### update : 2025/01 by ITRI (EOSL-R3)

## Ubuntu Operating System Installation

Before proceeding with this guide, you need to install Ubuntu operating system on your Genio EVK board:

1. **Download Ubuntu for MediaTek Genio**: Visit [https://ubuntu.com/download/mediatek-genio](https://ubuntu.com/download/mediatek-genio) to download the appropriate Ubuntu image for your specific Genio board (350, 510, 700, or 1200).

2. **Follow Official Installation Guide**: Please follow the official MediaTek installation tutorial at [https://mediatek.gitlab.io/genio/doc/ubuntu/](https://mediatek.gitlab.io/genio/doc/ubuntu/) for detailed flashing instructions.

> ### Prerequisites
> * A **Genio-350, 510, 700**, or **1200 EVK** with the Ubuntu operating system successfully flashed.
> * Ensure the **WiFi6** or **5G** module is connected to the antenna and activated.
> * It is recommended to use **Miniconda for Linux** or **Virtualenv** to manage Python APIs.


## NeuronRT
### Step 1: Install CMake from source code

```bash
$ sudo apt-get install libssl-dev
$ git clone https://github.com/Kitware/CMake.git && cd CMake
$ ./bootstrap && make && sudo make install && cd
```

### Step 2: Install and verify the NeuronRT library
Follow the [Instruction](https://mediatek.gitlab.io/genio/doc/ubuntu/bsp-installation/neuropilot.html#) to install and verify the NeuroPilot Hardware Packages (NeuronRT), to access MediaTek Deep Learning Accelerator (MDLA) and Vision Processor (VP).

## ArmNN

### Step 1: Download pre-built **ArmNN-linux-aarch64** library from [ArmNN-Release](https://github.com/ARM-software/armnn/releases) .

```bash
curl -L https://github.com/ARM-software/armnn/releases/download/v24.11/ArmNN-linux-aarch64.tar.gz | tar -xz -C ~/
```

### Step 2: Setup the environment veriable

```bash
$ vim ~/.bashrc         # add LD_LIBRARY_PATH=</path/to/ArmNN-linux-aarch64>:$LD_LIBRARY_PATH to .bashrc
$ source ~/.bashrc
```

## KleidiAI
### Step 1: Install KleidiAI from source code

```bash
$ git clone https://git.gitlab.arm.com/kleidi/kleidiai.git && cd kleidiai
$ cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/aarch64-none-linux-gnu.toolchain.cmake -S . -B build/
$ cmake --build ./build && cd
```

　

<br>
<div align="right">

<a href="https://github.com/R300-AI/ITRI-AI-Hub/tree/main/Model-Zoo"> 

[ Finished >> Run Demo ]

</div>
