# System Set Up
## WSL
WSL installation is simple with ```wsl --install```
## Cuda
You can install Cuda toolkit from the [Nvidia website](https://developer.nvidia.com/cuda-downloads).
From options chose
* Operating System: Linux
* Architecture: x86_64
* Distribution: WSL-Ubuntu
* Version: 2.0
## Conda
1. Go to the [Anaconda Archive](https://repo.anaconda.com/archive/) and find Linux distribution for your machine.
2. From WSL terminal run ```wget https://repo.continuum.io/archive/[DISTRIBUTION]```  
For example: ```wget https://repo.continuum.io/archive/Anaconda3-5.2.0-Linux-x86_64.sh```
3. Run the script with ```bash [DISTRIBUTION]```  
For example: ```bash Anaconda3-5.2.0-Linux-x86_64.sh```
4. If not done automatically, you can add it to path in .bashrc file.
5. With that, you automatically have python and pip installed.
## Conda environment
After cloning this repository, you can create a conda environment with the following command:  
```conda env create -f environment.yml```  
After running this command, pip will automatically install all the necessary packages.  
To activate the environment, run: ```conda activate bp_wsl```
# Dataset
The main dataset used in this project is [BioCreative-V-CDR-Corpus](https://github.com/JHnlp/BioCreative-V-CDR-Corpus/tree/master).  
For using using different datasets, you need to
1. Have your dataset in desired format. The dataset is plain text with entities annotated as \<category=