import math
import pandas as pd
import biovec
from sklearn.utils import class_weight

# get enzyme class from ec string
def getClassFromEC(ec):
	try:
		if math.isnan(ec):
			print(np.zeros((8,)))
			return 0
	except:
		ecval = int(str(ec).split('.')[0])
		return_y = np.zeros((8,))
		return_y[ecval] = 1.0
		print(ecval, return_y)
		return ecval

ds = pd.read_csv('data/protein_dataset_cdhit_80.csv')
seq = ds.iloc[:,6:7].values
ec = ds.iloc[:,5:6].values

# biovec model training
f = open("dataset.fasta", "w")
for i in range(len(seq)):
	f.write("> " + str(i))
	f.write("\n")
	f.write(seq[i][0])
	f.write("\n")

pv = biovec.models.ProtVec("dataset.fasta", corpus_fname="output_corpusfile_path.txt", n=3)
pv.save('enzyme_dataset_biovec.model')

pv = biovec.models.load_protvec('enzyme_dataset_biovec.model')
X = []
y = []
for i in range(len(seq)):
	print(i, len(seq))
	try:
		vec = pv.to_vecs(seq[i][0])
		vec = np.asarray(vec)
		print(vec.shape)
		if vec.shape == (3,100):
			y_val = getClassFromEC(ec[i][0])
			X.append(vec)
			y.append(y_val)
	except:
		pass

X = np.asarray(X)
y = np.asarray(y)
print(X.shape, y.shape)
pickle out
filename = 'X.pickle'
outfile = open(filename,'wb')
pickle.dump(X ,outfile)
outfile.close()
filename = 'y.pickle'
outfile = open(filename,'wb')
pickle.dump(y, outfile)
outfile.close()


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
