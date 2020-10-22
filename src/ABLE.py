# Libraries
import pandas as pd 
import numpy as np
import keras
import tensorflow as tf
from keras.models import Model
from keras.models import load_model
from keras.models import Sequential
import keras.backend as K
from keras import optimizers
from keras.layers import Dense, Dropout, LSTM, Input, Bidirectional, Flatten
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
from sklearn.utils import class_weight
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
from keras_self_attention import SeqSelfAttention

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

# k fold cross val
num_folds = 5
acc_per_fold = []
loss_per_fold = []
f1_per_fold = []
bal_acc_per_fold = []

fold_num = 1
kfold = KFold(n_splits=num_folds, shuffle=True)
for train, test in kfold.split(X, y):
    print("Fold Number: ", fold_num)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    # X_train, y_train = shuffle(X_train, y_train, random_state = 42)

    y_train = np_utils.to_categorical(y[train])
    y_test = np_utils.to_categorical(y[test])

    # Neural Network model
    inp = Input(shape=(3,100))
    x = Bidirectional(LSTM(128, activation = 'tanh', return_sequences = True))(inp)
    # x = Dropout(0.4)(x)
    x = Bidirectional(LSTM(128, activation = 'tanh', return_sequences = True))(x)
    # x = Dropout(0.7)(x)
    x = SeqSelfAttention(attention_activation='tanh')(x)
    x = Flatten()(x)
    # x = Dropout(0.7)(x)
    out = Dense(8, activation='softmax')(x)

    model = Model(inp, out)
    # print(model.summary())
    opt = keras.optimizers.Adam(learning_rate = 1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    mcp_save = keras.callbacks.callbacks.ModelCheckpoint('../models/Abilstm.h5', save_best_only=True, monitor='val_accuracy', verbose=1)
    reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    callbacks_list = [reduce_lr, mcp_save]

    history = model.fit(X[train], y_train, batch_size = 512, epochs = 700, validation_data = (X[test], y_test), shuffle = False, callbacks = callbacks_list)

    model = load_model('../models/Abilstm.h5', custom_objects=SeqSelfAttention.get_custom_objects())
    eval_ = model.evaluate(x = X[test], y = y_test)
    acc_per_fold.append(eval_[1])
    loss_per_fold.append(eval_[0])
    print("Loss: " + str(eval_[0]) + ", Accuracy: " + str(eval_[1]))
    print(eval_)

    y_pred = model.predict(X[test])

    # Metrics
    print("Confusion Matrix")
    matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    print(matrix)

    print("F1 Score")
    f1 = f1_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average = 'weighted')
    print(f1)
    f1_per_fold.append(f1)
    print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

    print("Balanced Accuracy Score")
    bal_acc = balanced_accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    print(bal_acc)
    bal_acc_per_fold.append(bal_acc)

    fold_num += 1

print("K Fold Metrics")
print("Accuracy: ", np.mean(acc_per_fold), " +- ", np.std(acc_per_fold))
print("Balanced Accuracy: ", np.mean(bal_acc_per_fold), " +- ", np.std(bal_acc_per_fold))
print("Weighted F1 Score: ", np.mean(f1_per_fold), " +- ", np.std(f1_per_fold))
print(acc_per_fold, bal_acc_per_fold, f1_per_fold)

'''K Fold Metrics
Accuracy:  0.9001388311386108  +-  0.008349277372240133
Balanced Accuracy:  0.7343931109466725  +-  0.035337177412446234
Weighted F1 Score:  0.8981577707658739  +-  0.00889887017359589
'''


