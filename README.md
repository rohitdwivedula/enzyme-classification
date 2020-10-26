[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# ABLE: Attention Based Learning for Enzyme classification

Attention based deep learning model to classify a given protein sequence into the seven classes of enzymes or a negative class. The code for the machine learning and deep learning models (with cross validation and sampling methods), along with the code for preprocessing (vectorization) and postprocessing (statistical testing and plotting) can be found in the `/src` folder. Run the machine learning models by running `python3 ml_models.py` in the directory. DL models can be run using the script `run_dl_models.py` which can be run by a command of the format:

```
python3 run_dl_models.py CNN --epochs 200 --batch 256
```

`epochs` and `batch` are optional arguments which default to `100` and `128` respectively, if not specified. The `model` used has to be the first argument to the script and has to be one of `CNN, LSTM, BILSTM, GRU, ABLE`. The `results` directory contains all information about the performance of the models - including runtimes, multiclass confusion matrices, f-score, precision, and recall, among other metrics, as both Python `pickle` files and CSVs. `results/DL` contains the training history for each run of the models, stored as `.npy` files - the syntax for each filename in this directory is `{{MODEL_NAME}}_{{SAMPLING_METHOD}}_{{TESTING_FOLD}}_{{NUM_EPOCHS}}_{{BATCH_SIZE}}.npy`. 