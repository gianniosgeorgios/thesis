# Control Robot Vehicle using Speech Recognition techniques

Author: Giannios Georgios-Taxiarchis

Supervisor: A.G Stafylopatis

Approved (26/10/2021) by: A.G Stafylopatis, G. Stamou, S. Kollias 

Lab: Artificial Intelligence and Learning Systems Laboratory (AILS)


## Thesis Description 

The object of the present diploma thesis is the design, implementation and evaluation of a system that will contribute to the interaction between a person and a robotic vehicle. This interaction will be executed using specific voice commands (eg stop, go, right ...), through which the direction of the vehicle will be controlled. For a detailed description, please read thesis report (in Greek) [here](http://artemis.cslab.ece.ntua.gr:8080/jspui/bitstream/123456789/18128/1/Thesis_Giorgos_Giannios.pdf) or check demo in this [video](https://www.youtube.com/watch?v=nEsMhEaqhxc). 

![use](https://user-images.githubusercontent.com/50829499/140767728-040f1db6-625c-47c5-b7a3-a4e49bf2b925.png)


## Setup

The experinments about feature extraction process were conducted in Time Domain (`codes/Time_Domain_Analysis.ipynb`) and Frequency Domain (`codes/Frequency_Domain_Analysis.ipynb`). You can run these notebooks in your broswer using Google Colaboratory Environment (GPU is required). So download .ipynb files and upload them to a Google Colab folder named `my_project` (in home dir of Colab).

Note: Each Notebook is organized to contents, in comments in almost each cell.

## Evaluate Results of thesis
 
In order to evaluate results, that are present in diploma thesis (plots, matrices, accuracy values etc ...):

The first step is to launch Google Speech Commands Dataset. You can download it from [here](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz). The commands that we use are 12 ('yes', 'no', 'go', 'stop', 'right', 'left', 'one', 'two', 'three', 'four', 'down', 'up'). So, inside 'my_project' folder, upload a folder named speech_commands_v0.01, that contains /yes, /no ... /up subfolders. Example:

```
/my_project
├── speech_commands_v0.01/
│   ├── yes/
│   ├── no/
│   ├── stop/
...
│   ├── four/
├── Time_Domain_Analysis.ipynb/
├── Frequency_Domain_Analysis.ipynb/
```

Now you can run either `Time_Domain_Analysis.ipynb` or `Frequency_Domain_Analysis.ipynb` to evaluate results.

## Run demo, using your voice

In order to run demo (speech recognizer using your voice), follow the next steps:

1. Download trained model (ResNet.zip)
2. Download `evaluate.py` code
3. Install requirenments  

```
pip install -r requirenments.txt
```

4. Run evaluate script

```
python3 evaluate.py
```








