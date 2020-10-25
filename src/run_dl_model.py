import argparse
import pandas as pd 
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import time

# check if user parameters are valid
parser = argparse.ArgumentParser(description='Run one DL models on protein dataset.')
parser.add_argument('model', help='Name of the model to run it on. Must be one of CNN, GRU, LSTM, BILSTM, ABLE')
args = parser.parse_args()

if args.model not in ["CNN", "GRU", "LSTM", "BILSTM", "ABLE"]:
	print("Model", args.model, "is not defined. Please make changes to dl_models.py and this file")
	exit(0)

## Keras/Tensorflow Imports and Setup
import keras
import tensorflow as tf
from keras.models import Model
from tensorflow.keras.models import load_model
from keras.utils import to_categorical, np_utils
from keras.regularizers import l2
from tensorflow.compat.v1 import ConfigProto # GPU

tf.keras.backend.clear_session()
config = ConfigProto()
config.gpu_options.allow_growth = True
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

LIMIT = 3 * 1024
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=LIMIT)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

# import all the neural network models
from dl_models import get_dl_model 

# Load Dataset
with open('../data/X.pickle','rb') as infile:
	X = pickle.load(infile)

with open('../data/y.pickle','rb') as infile:
	y = pickle.load(infile)

X = X[y != 7]
y = y[y != 7]

kf = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
SAMPLING_METHODS = ["NONE", "SMOTE"]
fold = 1

all_results = []

for train_index, test_index in kf.split(X, y):
	print("Beginning run on fold #", fold)
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	y_train = np_utils.to_categorical(y_train)
	y_test = np_utils.to_categorical(y_test)
	
	for sampling_method in SAMPLING_METHODS:

		if sampling_method == "NONE":
			X_resampled, y_resampled = X_train, y_train
		elif sampling_method == "SMOTE":
			X_train = np.reshape(X_train, newshape=(X_train.shape[0], 300))
			start_smote = time.time()
			print("Sampling with SMOTE begin")
			X_resampled, y_resampled = SMOTE(random_state=1).fit_resample(X_train, y_train)
			X_resampled = np.reshape(X_resampled, newshape = (X_resampled.shape[0], 3, 100))
			print("SMOTE complete in %.2f seconds"%(time.time() - start_smote))

		start_train = time.time()
		model = get_dl_model(args.model)
		print(model.summary())
		opt = keras.optimizers.Adam(learning_rate = 1e-3)
		model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
		mcp_save = keras.callbacks.ModelCheckpoint('models/saved_model.h5', save_best_only=True, monitor='val_accuracy', verbose=1)
		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
		callbacks_list = [reduce_lr, mcp_save]
		history = model.fit(X_train, y_train, batch_size = 256, epochs = 200, validation_split = 0.1, shuffle = True, callbacks = callbacks_list)
		model = load_model('models/saved_model.h5')
		end_train = time.time()
		
		y_pred = model.predict(X_test)
		# Metrics
		results_dict = {
			"model": args.model,
			"fold": fold,
			"sampling": sampling_method, 
			"confusion_matrix": confusion_matrix(y_test, y_pred),
			"report": classification_report(y_test, y_pred, output_dict = True),
			"train_time": end_train - start_train,
			"test_time": end_test - end_train
		}
		end_test = time.time()
		
		np.save("../results/" + args.model + "_" + sampling_method + "_" + str(fold) + ".npy", history.history)
		print("Model", args.model, "done running on fold", fold, "with sampling strategy", sampling_method, "in", round(time.time()-start_train, 3), "s")
		all_results.append(results_dict)
		with open('dl_results.pickle', 'wb') as handle:
			pickle.dump(all_results, handle)
		print("Model", model_name, "on fold", fold, "with sampling strategy", sampling_method, "completed in total of", time.time()-start_train, "seconds")

	fold += 1