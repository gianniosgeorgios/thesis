# Control Robot Vehicle using Speech Recognition techniques

## Abstract 

The object of the present diploma thesis is the design, implementation and evaluation of a system that will contribute to the interaction between a person and a robotic vehicle. This interaction will be executed using specific voice commands (eg ‘stop’, ‘go’, ‘right’ ...), through which the direction of the vehicle will be controlled.

Initially, some audio recordings were assembled for each of the commands. These samples were then converted into appropriate representation to be fed into a deep machine learning model. The experimentation concerned both the extraction of features and the synthesis of the neural network architecture.

Regarding feature extraction, a time-field experiment was first conducted with each acoustic signal being examined as a time series.  Some weaknesses quickly emerged, which made it imperative to switch to the frequency domain. In this field, after experimentation, the model has emerged, as well as features that achieve the highest accuracy. The final model that was created is available and can be tested on acoustic signals, and even in real time.
