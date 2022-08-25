# Notebooks

These are a few notebooks intended to be used on Google Colab. They will clone this repo and install the dependencies required to run it on Colab. Make sure you're running them with a GPU-accelerated runtime. (I have not tested them locally, but if you're running on a local machine, you probably don't need them.)

If you want data, you will need to provide your own dataset. I have included a block that will extract the data from shared Google Drive links if they are provided. More instructions are included in the notebooks.

## Mask Demo [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/d-wang-0/MASK_public/blob/colab/notebooks/Mask_Demo.ipynb)

This notebook is intended to serve as a basic demonstration of the NER models. You can load, train and evaluate various models with a few adjustable parameters.

## Mask Dev [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/d-wang-0/MASK_public/blob/colab/notebooks/Mask_Dev.ipynb)

This notebook is intended to serve as more of a development notebook for the NER models. Most of the code required to run the `NER_BiLSTM_ELMo_i2b2_pad_mask` model is directly included in the notebook rather than relying on imports for an easier development workflow on Colab.

Some basic documentation of the code is included. For more details about the model see [the wiki](wiki/BiLSTM-ELMo#modified).

## Mask Dataset Exploration [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/d-wang-0/MASK_public/blob/colab/notebooks/Mask_Dataset_Exploration.ipynb)

This notebook is intended to explore the dataset and generate some statistics. This notebook should also run fine locally without access to a GPU (just skip the setup and probably move it into the base directory).
