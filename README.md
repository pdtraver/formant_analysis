# formant_analysis

This library is part of a larger testing framework for formant analysis developed at the BITS lab at NYU. The relevant portions to the PTSA final project are in the classes.py file, namely the WLP class and BufferedAudio class, which generates the QCP windows. GCI estimates were obtained using the FCN_GCI from [Ardaillon et. al. 2020]. To recreate the project, the DeepFormants, GCI_FCN and ftrack repos are also needed, which can be found here: \

DeepFormants: https://github.com/MLSpeech/DeepFormants \
FCN_GCI: https://github.com/ardaillon/FCN_GCI \
ftrack: https://github.com/njaygowda/ftrack/tree/master/ftrack_tvwlp_v1 \