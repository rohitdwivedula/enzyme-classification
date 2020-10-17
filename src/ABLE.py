# Libraries
import biovec
import pandas as pd 
import numpy as np
import math
import keras
import tensorflow as tf
from keras.models import Model
from tensorflow.keras.models import load_model
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

# get enzyme class from ec string
# def getClassFromEC(ec):
# 	print(ec)
# 	try:
# 		if math.isnan(ec):
# 			print(np.zeros((8,)))
# 			return 0
# 	except:
# 		ecval = int(str(ec).split('.')[0])
# 		return_y = np.zeros((8,))
# 		return_y[ecval] = 1.0
# 		print(ecval, return_y)

# 		return ecval

# 	# return np.zeros((8,))

ds = pd.read_csv('data/protein_dataset_cdhit_80.csv')
seq = ds.iloc[:,6:7].values
ec = ds.iloc[:,5:6].values

# biovec model training
# f = open("dataset.fasta", "w")
# for i in range(len(seq)):
# 	f.write("> " + str(i))
# 	f.write("\n")
# 	f.write(seq[i][0])
# 	f.write("\n")

# pv = biovec.models.ProtVec("dataset.fasta", corpus_fname="output_corpusfile_path.txt", n=3)
# pv.save('enzyme_dataset_biovec.model')

# # # load pretrained model
# pv = biovec.models.load_protvec('enzyme_dataset_biovec.model')

# X = []
# y = []

# for i in range(len(seq)):
# 	print(i, len(seq))
# 	try:
# 		vec = pv.to_vecs(seq[i][0])
# 		vec = np.asarray(vec)
# 		print(vec.shape)
# 		if vec.shape == (3,100):
# 			y_val = getClassFromEC(ec[i][0])
# 			X.append(vec)
# 			y.append(y_val)
# 	except:
# 		pass

# # train test split
# X = np.asarray(X)
# y = np.asarray(y)
# print(X.shape, y.shape)

# pickle out
# filename = 'X.pickle'
# outfile = open(filename,'wb')
# pickle.dump(X ,outfile)
# outfile.close()

# filename = 'y.pickle'
# outfile = open(filename,'wb')
# pickle.dump(y ,outfile)
# outfile.close()

# pickle in
infile = open('data/X.pickle','rb')
X = pickle.load(infile)
infile.close()

infile = open('data/y.pickle','rb')
y = pickle.load(infile)
infile.close()

df = pd.DataFrame(y)
df.columns = ["enz"]
print(df["enz"].value_counts())

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y),
                                                 y)
print(class_weights)
# weights = {0:0.7, 1:10.20713346, 2:8.53112995, 3:8.46352015, 4:40.62895616, 5:80.47985372, 6:80.97642173, 7:1.385274}

# def create_class_weight(labels_dict,mu=0.15):
#     total = 0
#     keys = labels_dict.keys()
#     for i in keys:
#     	total += labels_dict[i]
#     class_weight = dict()

#     for key in keys:
#     	print(total)
#     	val1 = mu*total
#     	val2 = float(labels_dict[key])
#     	print(val1, val2)
#     	score = math.log(val1/val2)
#     	class_weight[key] = score if score > 1.0 else 1.0

#     return class_weight

# # random labels_dict
# labels_dict = {0: 91836, 1: 7223, 2: 10412, 3: 10893, 4: 3444, 5: 1880, 6: 1776, 7: 73}

# weights = dict(enumerate(create_class_weight(labels_dict)))
# print(weights)

# y_arg = []
# for i in y:
# 	y_arg.append(np.argmax(i))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
# X_train, y_train = shuffle(X_train, y_train, random_state = 42)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

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
print(model.summary())
opt = keras.optimizers.Adam(learning_rate = 1e-3)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

mcp_save = keras.callbacks.callbacks.ModelCheckpoint('models/Abilstm.h5', save_best_only=True, monitor='val_accuracy', verbose=1)
reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
callbacks_list = [reduce_lr, mcp_save]

history = model.fit(X_train, y_train, batch_size = 512, epochs = 200, validation_data = (X_test, y_test), shuffle = False, callbacks = callbacks_list)

# model = load_model('models/Abilstm.h5', custom_objects=SeqSelfAttention.get_custom_objects())
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

'''Run 1: 200 epochs
2 Layers (32 units each) + 0.4 Dropout below
Loss: 0.6390913128852844, Accuracy: 0.8071155548095703
'''


