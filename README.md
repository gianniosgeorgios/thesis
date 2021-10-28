# Control Robot Vehicle using Speech Recognition techniques

Author: Giannios Georgios-Taxiarchis
Lab: Artificial Intelligence and Learning Systems Laboratory (AILS Lab)
Supervisor: A.G Stafylopatis


## Thesis Description 

The object of the present diploma thesis is the design, implementation and evaluation of a system that will contribute to the interaction between a person and a robotic vehicle. This interaction will be executed using specific voice commands (eg stop, go, right ...), through which the direction of the vehicle will be controlled.

## Setup

The experinments about feature extraction process were conducted in Time Domain () and Frequency Domain (). You can run these notebooks in your broswer using Google Colaboratory environment. So download .ipynb files and upload them to a Google Colab folder named 'my_project' (in home dir of Colab).


Note: Each Notebook is organized to contents, in comments in almost each cell.

## Evaluate Results of thesis
 
In order to evaluate results, that are present in diploma thesis, (plots, matrices, accuracy values etc ...):

The first step is to launch Google Speech Commands Dataset. You can download it from [here](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz). The commands that we use are 12 ('yes', 'no', 'go', 'stop', 'right', 'left', 'one', 'two', 'three', 'four', 'down', 'up'). So, inside 'my_project' folder, upload a folder named speech_commands_v0.01, that contains /yes, /no ... /up subfolders.

## Run demo, using your voice

In order to run demo (speech recognizer using your voice):

The first step is to download a)model, (provided in ResNet.zip) and b) pipeline code (provided in evaluate.py). 





