## Overview
This repository provides demo implementation accompanying the paper [Vaem: a deep generative model for heterogeneous mixed type data](https://papers.nips.cc/paper/2020/file/8171ac2c5544a5cb54ac0f38bf477af4-Paper.pdf).  The code for VAEM can be found in `Main_Notebook.ipynb`, which demonstrates data preprocessing, load and train VAEM model, and performing down stream tasks. 

## Dependencies
* Numpy (1.15.2)
* Tensorflow (1.4.0)
* Scipy (1.1.0)
* Sklearn (0.20.0)
* Matplotlib (3.0.0)
* Pandas (0.23.4)
* Seaborn (0.9.0)

## File structure
* `data`: contains dataset in this demo. 
  * `bank`: This folder is supposed to store [Bank Marketing dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing). Note that due to compliance issue, we are not able to provide any dataset here. 
  An example of data format is provided in the .csv file (it does not contain any real data). If you are running the active learning task, note that the target variable of active learning needs to be placed at the last column of the data matrix. 
And we require the variable names to be put in the first row of the data file.
* `models`: contains deconder/encoder/vame model definitions
  * `decoders.py`: implements decoder models used in VAEM
  * `encoders.py`: implements encoder models (for marginal VAEs and dependenciy VAEs) used in VAEM
  * `model.py`: implements VAEM model
* `hyperparameters`: this folder contains `.json` files that stores configurations for the model
* `saved_weights`: This is where TF model is stored. Unfortunately, we cannot provide pretrained tensorflow models on bank dataset. 
* `utils`: some utility functions used by VAEM.
  * `process`: functions used in data preprocessing
  * `reward`: reward estimation for sequential active information acquisition
  * `active_learning`: functions that trains VAEM model and (optionally) performs sequential active information acquisition.
* `Main_Notebook.ipynb`,Jupyter notebook demonstrates data preprocessing, load and train VAEM model, and performing down stream tasks. 

## Usage
To run the demo, you need to first download the [Bank Marketing UCI dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing), 
and put the csv file under `data/bank`. You will need to preprocess the data into the format according to our example .csv 
file (which does not contain any real data). This can done by splitting the text into columns using `;` as delimiters. Then,
simply run `Main_Notebook.ipynb`. This notebook train/or load a VAEM model on Bank dataset, 
and demonstrates how to perform sequential active information acquisition (SAIA) and imputation. 
By default, it trains a new model on Bank dataset. If you would like to load a pre-trained model, by default it will load a pre-trained tensorflow model from `saved_weights/bank/`. Note that in order to perform 
active information acquisition, an additional third stage training is required. This will add a discriminator (predictor)
to the model, which is required for SAIA. The configurations for VAEM can be found in `.json` files in `hyperparameters/bank`, 
which include:
* "list_stage" : list of stages that you would like the model to be trained. stage 1 = training marginal VAEs, stage 2 = training dependency network,  stage 3 = add predictor and improve predictive performance. The default is [1,2]. 
* "epochs" : number of epochs for training VAEM. If you would like to load a pretrained model rather than training a new one, you can simply set this to zero.
* "latent_dim" : size of latent dimensions of dependency network, 
* "p" : upper bound for artificial missingness probability. For example, if set to 0.9, then during each training epoch, the algorithm will randomly choose a probability smaller than 0.9, and randomly drops observations according to this probability. Our suggestion is that if original dataset already contains missing data, you can just set p to 0. 
* "iteration" : iterations (number of mini batches) used per epoch. set to -1 to run the full epoch. If your dataset is large, please set to other values such as 10.
* "batch_size" : iterations (number of mini batches) used per epoch. set to -1 to run the full epoch. If your dataset is large, please set to other values such as 10.
* "K" : the dimension of the feature map (h) dimension of PNP encoder.
* "M" : Number of MC samples when perform imputing. 
* "repeat" : number of repeats.
* "data_name" : name of the dataset being used. Our default is "bank".
* "output_dir" : Directory where the model is stored. Our default is "./saved_weights/bank/",
* "data_dir" : Directory where the data is stored. Our default is "./data/bank/",
* "list_strategy" : list of strategies for active learning, 0 = random, 1 = single ordering. Default: [1]


## Citing our paper

If you found our work useful in your research, please consider citing

```
@article{ma2020vaem,
  title={VAEM: a Deep Generative Model for Heterogeneous Mixed Type Data},
  author={Ma, Chao and Tschiatschek, Sebastian and Turner, Richard and Hern{\'a}ndez-Lobato, Jos{\'e} Miguel and Zhang, Cheng},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```

## Credit and note

Many of the code is adapted from [EDDI](https://github.com/microsoft/EDDI/). We encourage you to look into this repository if you are interested in sequential active feature acquisition with VAEs. 

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Disclaimer

This is not an official Microsoft product.
