# System Set Up
## WSL
(skip if on linux)
WSL installation is simple with ```wsl --install```
## Cuda
You can install Cuda toolkit from the [Nvidia website](https://developer.nvidia.com/cuda-downloads).
From options chose
* Operating System: Linux
* Architecture: x86_64
* Distribution: WSL-Ubuntu
* Version: 2.0
After, run the commands shown on the website.
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
Optionally you can run setup.sh included into repository that does the necessary steps automatically.
# Dataset
The main dataset used in this project is [BioCreative-V-CDR-Corpus](https://github.com/JHnlp/BioCreative-V-CDR-Corpus/tree/master).  
For using using different datasets, you need to
1. Have your dataset in desired XML-style format. The dataset is plain text with entities annotated as 
```Text that isn't an entity and entities are <category="CategoryName">EntityName</category>.```  
In other words, the entities have to be enclosed in the category tag with category name. The text cannot contain the category tag or
Phi-3 mini special tokens which are:  
```<unk>, <s>, </s>, <|endoftext|>, <|end|>, <|assistant|>, <|user|>, <|system|>, <|placeholder1|> to <|placeholder6|>```
2. Format the dataset using the ```format_dataset.py``` script. Use filename as script parameters. Example:  
```python format_dataset.py train_dataset.txt test_dataset.txt dev_dataset.txt```
3. In ```system_prompt.txt``` change the category names to the category names in your dataset and provide your examples instead of the examples provided.
4. In ```finetune.py``` script, change the dataset path to the path of your dataset.
