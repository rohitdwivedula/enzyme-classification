# Libraries
import pandas as pd 
import numpy as np
import keras
import tensorflow as tf
from keras.models import Model
from tensorflow.keras.models import load_model
from keras.models import Sequential
import keras.backend as K
from keras import optimizers
from keras.layers import Dense, Dropout, Input, Conv1D, Flatten
from keras.utils import to_categorical, np_utils
from keras.regularizers import l2
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
import pickle
from keras import regularizers
from keras import backend as K

# GPU
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

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


# pickle in
with open('../data/X.pickle','rb') as infile:
	X = pickle.load(infile)

with open('../data/y.pickle','rb') as infile:
	y = pickle.load(infile)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
# X_train, y_train = shuffle(X_train, y_train, random_state = 42)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Neural Network model
inp = Input(shape=(3,100))
x = Conv1D(128, (3), padding = 'same', activation = 'relu')(inp)
x = Conv1D(64, (3), padding = 'same', activation = 'relu')(x)
x = Conv1D(32, (3), padding = 'same', activation = 'relu')(x)
x = Flatten()(x)
# x = Dropout(0.2)(x)
x = Dense(32, activation = 'relu')(x)
# x = Dropout(0.7)(x)
out = Dense(8, activation='softmax')(x)

model = Model(inp, out)
print(model.summary())
opt = keras.optimizers.Adam(learning_rate = 1e-3)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

mcp_save = keras.callbacks.callbacks.ModelCheckpoint('models/cnn.h5', save_best_only=True, monitor='val_accuracy', verbose=1)
reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
callbacks_list = [reduce_lr, mcp_save]

history = model.fit(X_train, y_train, batch_size = 256, epochs = 200, validation_data = (X_test, y_test), shuffle = False, callbacks = callbacks_list)

model = load_model('models/cnn.h5')
eval_ = model.evaluate(x = X_test, y = y_test)
print("Loss: " + str(eval_[0]) + ", Accuracy: " + str(eval_[1]))
print(eval_)

y_pred = model.predict(X_test)

# Metrics
print("Confusion Matrix")
matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print(matrix)

print("F1 Score")
print(f1_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average = 'weighted'))

print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

'''
Run 1: 200 epochs
3 Layer CNN (128, 64, 32 units)
2 Layer Dense (32, 8)
Loss: 1.0134365558624268, Accuracy: 0.8421836495399475
'''