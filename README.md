# Control Robot Vehicle using Speech Recognition techniques

## Thesis Description 

The object of the present diploma thesis is the design, implementation and evaluation of a system that will contribute to the interaction between a person and a robotic vehicle. This interaction will be executed using specific voice commands (eg stop, go, right ...), through which the direction of the vehicle will be controlled.

## Setup

The experinments about feature extraction process were conducted in Time Domain () and Frequency Domain (). You can run these notebooks in your broswer using Google Colaboratory environment. So download .ipynb files and upload them to a Google Colab folder named 'my_project' (in home dir of Colab)

## Evaluate Results of thesis 

The first step is to launch Google Speech Commands Dataset. You can download it from [here](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz). The commands that we use are 12 ('yes', 'no', 'go', 'stop', 'right', 'left', 'one', 'two', 'three', 'four', 'down', 'up'). So, inside 'my_project' folder, upload a folder named speech_commands_v0.01, that contains /yes, /no ... /up subfolders.

The second step is to change paths, according to where your dataset is placed.

The first step, in order to evaluate results, that are present in diploma thesis, (plots, matrices, accuracy values etc ...), is to launch dataset. 









Initially, some audio recordings were assembled for each of the commands. These samples were then converted into appropriate representation to be fed into a deep machine learning model. The experimentation concerned both the extraction of features and the synthesis of the neural network architecture.

Regarding feature extraction, a time-field experiment was first conducted with each acoustic signal being examined as a time series.  Some weaknesses quickly emerged, which made it imperative to switch to the frequency domain. In this field, after experimentation, the model has emerged, as well as features that achieve the highest accuracy. The final model that was created is available and can be tested on acoustic signals, and even in real time.
