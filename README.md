# Installation Instructions

1. Download [Miniconda3](https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh), install the miniconda using this command : 
```bash
bash Miniconda3-py310_23.5.2-0-Linux-x86_64.sh
```
2. Make sure to follow the instruction and pay attention to installation folder, then try to activate the conda.
```bash
source /root/miniconda3/bin/activate
```
3. Create the environment for this parser to work
```bash
conda create -n amr_parser python=3.7
```
4. After creating the environment, activate it
```bash
conda activate amr_parser
```
5. Install the torch dependency used for this parser
```bash
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```
6. Install the rest of pip dependency
```bash
pip install -r pip_requirements.txt
```
7. Open python from console, and run this to download model for preprocessing, make sure use the python 
interpreter that is inside the conda environment
```python
import nltk
nltk.download('punkt')
```
8. Download these trained models for parsing purpose
Trained AMR Parser Bahasa Indonesia : [trained_model](https://storage.googleapis.com/amr-ta2-bucket/runpod-folder/indonesia-amr-parser/pretrained_model_indonesia/epoch140.pt)

# Usage Instructions
1. Download vocabs for parsing purpose, these vocab can be modified later on based on the 
training dataset used at training phase
vocabs : [vocabs](https://storage.googleapis.com/amr-ta2-bucket/runpod-folder/indonesia-amr-parser/pretrained_model_indonesia/vocabs.zip)

2. Create a folder and put both the vocabs and the trained model that is dowloaded earlier in there. 
 To make the instruction clearer, we will name it trained_model.

3. Direct the console to `gsii_parser/scripts/30_Model_Parser_GSII` 
There are still some improvement to be made, so the variables have to be changed manually from the
source code.

4. There are still some limitations, so the sentence parsed still hardcoded to the source code. It's on `Generate_Sentence.py`

5. After typing the sentence that want to be parsed into AMR graph, we can run the file by typing 
in console (**make sure the console on the right path**),

```bash
python Generate_Sentence.py
```

# TRAINING 
## Annotate Dataset
**TBD**
## Create Vocabulary
**TBD**
## Managing Configuration
**TBD**
## Training The Model
**TBD**
## Fine-grained metric evaluation
Although for each `x` epochs (configurable) the model be evaluated  
**TBD**