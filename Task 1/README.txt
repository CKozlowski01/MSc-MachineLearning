This is a binary classifier for visual-question answering, where the 
inputs are images and text-based features, and the outputs (denoted as
yes=[1,0] and no=[0,1]) correspond (predicted) answers. This baseline
classifier makes use of two strands of features: (1) the first are 
produced by a CNN-classifier, and (2) the second are derived offline 
from a sentence embedding generator. The latter have the advantage of
being generated once, which can accelerate training due to being 
pre-training and loaded at runtime. Those two strands of features are
concatenated at trianing time to form a multimodal set of features, 
combining learnt image features and pre-trained sentence features.
See the following file for a description of the neural architecture 
of this classifier: model_arch_vqa_baseline.png

To run this Python program you will need to install the following:
conda install -c conda-forge pycocotools
pip install -q tf-models-official==2.10.0
pip install -q tensorflow==2.10
pip install -q tensorflow-text==2.10.*
pip install einops

Once you have installed the dependencies and downloaded the data (from
Blackboard), please update IMAGES_PATH accordingly.

Make sure you run this program where the following file exists: 
vaq2.cmp9137.sentence_transformers.txt

This program has been tested using an Anaconda environment with Python 3.9
on Windows 11 and Linux Ubuntu 22. You can run this program in two ways:
(1) from the command line> python VQAClassifier_baseline.py
(2) from an IDE such as Spyder, if that is compatible with the dependencies.

An example output of this program is provided in example_output.log.txt

Feel free to use, and modify, this program as part of your CMP9137 assignment.

If you have any questions, get in touch with the delivery team at:
hcuayahuitl@lincoln.ac.uk
lzhang@lincoln.ac.uk
jugao@lincoln.ac.uk
