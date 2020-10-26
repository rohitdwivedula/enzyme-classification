import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Conv1D, Bidirectional, GRU
from keras import backend as K

# GPU
from tensorflow.compat.v1 import ConfigProto
from keras_self_attention import SeqSelfAttention

def get_dl_model(model_name):
	if model_name == "CNN":
		inp = Input(shape=(3,100))
		x = Conv1D(128, (3), padding = 'same', activation = 'relu')(inp)
		x = Conv1D(64, (3), padding = 'same', activation = 'relu')(x)
		x = Conv1D(32, (3), padding = 'same', activation = 'relu')(x)
		x = Flatten()(x)
		x = Dense(32, activation = 'relu')(x)
		out = Dense(7, activation='softmax')(x)
		model = Model(inp, out)
		
	elif model_name == "LSTM":
		inp = Input(shape=(3,100))
		x = LSTM(32, activation = 'tanh', return_sequences = True)(inp)
		x = Dropout(0.2)(x)
		x = LSTM(32, activation = 'tanh')(x)
		out = Dense(7, activation='softmax')(x)
		model = Model(inp, out)
	
	elif model_name == "BILSTM":
		inp = Input(shape=(3,100))
		x = Bidirectional(LSTM(32, activation = 'tanh', return_sequences = True))(inp)
		x = Dropout(0.4)(x)
		x = Bidirectional(LSTM(32, activation = 'tanh'))(x)
		out = Dense(7, activation='softmax')(x)
		model = Model(inp, out)
	
	elif model_name == "GRU":
		inp = Input(shape=(3,100))
		x = GRU(32, activation = 'tanh', return_sequences = True)(inp)
		x = GRU(32, activation = 'tanh')(x)
		out = Dense(7, activation='softmax')(x)
		model = Model(inp, out)

	elif model_name == "ABLE":
		inp = Input(shape=(3,100))
		x = Bidirectional(LSTM(128, activation = 'tanh', return_sequences = True))(inp)
		x = Bidirectional(LSTM(128, activation = 'tanh', return_sequences = True))(x)
		x = SeqSelfAttention(attention_activation='tanh')(x)
		x = Flatten()(x)
		out = Dense(7, activation='softmax')(x)
		model = Model(inp, out)

	return model