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
```conda create -n <project_name> python=3.12```  
After running this command, pip will automatically install all the necessary packages.  
To activate the environment, run: ```conda activate <project_name>```  
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
# Training
Training is done in ```model_finetuning``` ipynb.
For models using the same tokenizer as ```Phi-3 mini```, you can use the same training script with only modifying the ```config.json``` file with your model name and your datasets.
For models using different tokenizers, you can use the same training script, but may need to modify the ```apply_chat_template``` function as they may annotate system/user/assistant prompts differently or may use different functions to apply them.
You may also reprint trainable modules in the model class, for which you run the cell right above LoRA configuration and then modify them in the LoRA configuration cell.
# Evaluation
Evaluation is done in ```evaluation.ipynb``` notebook.  
To change configurations of your testing, you can change the parameters in the ```config.json``` file.
You can also use multiple configurations file with ```load_config(<path>)``` function in the evaluation notebook before desired operations.
If using different models, follow the same format as the existing setups.
# Hyperparameter sweeps
For running hyperparameter sweeps, you can use the ```wandb_sweep.py``` script.
It's essentially a copy of the training script with only slight modifications, meaning the same conditions apply.
You can modify the sweep parameters in the ```sweep_config.yaml``` file.
To create the sweep run ```wandb sweep sweep_config.yaml```.
It will print a command you should run to start an agent for the sweep, running the script.
For multi-GPU setup, prefix the command with ```CUDA_VISIBLE_DEVICES=<GPU number>``` to run multiple agents in parallel, as our approach utilizes only 1 GPU at a time.