[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![DOI](https://zenodo.org/badge/290200950.svg)](https://zenodo.org/badge/latestdoi/290200950)



# ABLE: Attention Based Learning for Enzyme classification

Attention based deep learning model to classify a given protein sequence into the seven classes of enzymes or a negative class. The code for the machine learning and deep learning models (with cross validation and sampling methods), along with the code for preprocessing (vectorization) and postprocessing (statistical testing and plotting) can be found in the `/src` folder. Run the machine learning models by running `python3 ml_models.py` in the directory. DL models can be run using the script `run_dl_models.py` which can be run by a command of the format:

```
python3 run_dl_models.py CNN --epochs 200 --batch 256
```

`epochs` and `batch` are optional arguments which default to `100` and `128` respectively, if not specified. The `model` used has to be the first argument to the script and has to be one of `CNN, LSTM, BILSTM, GRU, ABLE`. 

Data used is provided in the `data.zip` file (64MB) - containing two pickle files (X.pickle and y.pickle), which contain the vectorized representations of the data. `X.pickle` contains an array of shape `(127537, 3, 100)` - 1,27,537 proteins represented in vectorized form of size `(3, 100)`. `Y.pickle` contains the labels for these data (numbers 0 to 7), with 0 representing the negative class. Before running any of the code in `src/`, make sure to extract `data.zip` in the root of this repository to create the `data/` directory with these two files.


The `results` directory contains all information about the performance of the models - including runtimes, multiclass confusion matrices, f-score, precision, and recall, among other metrics, as both Python `pickle` files and CSVs. 

* `results/dl` contains the training history for each run of the models, stored as `.npy` files - the syntax for each filename in this directory is `{{MODEL_NAME}}_{{SAMPLING_METHOD}}_{{TESTING_FOLD}}_{{NUM_EPOCHS}}_{{BATCH_SIZE}}.npy`. 
* Performance metrics of the runs are stored in these pickle files: `ABLE_results.pickle  BILSTM_results.pickle  CNN_results.pickle  GRU_results.pickle  LSTM_results.pickle  ML_results.pickle`, in the `results directory`
* The Jupyter Notebook in `results/postprocessing` contains the scripts to save Wilcoxin Signed Rank Test results between all pairs of models to files, on all performance metrics (precision, recall, f-score, balanced accuracy).

If you find these code or results useful in your research, please consider citing:

```
@article{ABLE2021,
    title = {ABLE: Attention Based Learning for Enzyme Classification},
    journal = {Computational Biology and Chemistry},
    pages = {107558},
    year = {2021},
    issn = {1476-9271},
    doi = {https://doi.org/10.1016/j.compbiolchem.2021.107558},
    url = {https://www.sciencedirect.com/science/article/pii/S1476927121001250},
    author = {Nallapareddy Mohan Vamsi and {Rohit Dwivedula}},
}

A [preprint of this article](https://www.biorxiv.org/content/early/2020/11/13/2020.11.12.380246) is also available on BioRxiv, though the preprint does not contain results from ADASYN sampling and comparisions with DeepEC (these additions were made during the peer review process). 
