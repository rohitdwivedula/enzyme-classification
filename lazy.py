import sys
from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

# pickle in
infile = open('X.pickle','rb')
X = pickle.load(infile)
infile.close()

X = np.asarray(X)
X_new = []
for i in X:
	X_new.append(np.reshape(i, (300,)))

X_new = np.asarray(X_new)

infile = open('y.pickle','rb')
y = pickle.load(infile)
infile.close()

X_train, X_test, y_train, y_test = train_test_split(X_new, y,test_size=0.4,random_state = 42)
clf = LazyClassifier(verbose=1, ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)

print(model)

'''
  0%|                                                    | 0/28 [00:00<?, ?it/s]{'Model': 'AdaBoostClassifier', 'Accuracy': 0.7055963932176811, 'Balanced Accuracy': 0.1229354375538883, 'ROC AUC': None, 'F1 Score': 0.5945753616640311, 'Time taken': 170.98004746437073}
  4%|█▍                                       | 1/28 [02:50<1:16:56, 170.98s/it]{'Model': 'BaggingClassifier', 'Accuracy': 0.9058708223071645, 'Balanced Accuracy': 0.6974640862988082, 'ROC AUC': None, 'F1 Score': 0.9007398954294045, 'Time taken': 467.53409147262573}
  7%|██▉                                      | 2/28 [10:38<1:52:38, 259.95s/it]{'Model': 'BernoulliNB', 'Accuracy': 0.5132216014897579, 'Balanced Accuracy': 0.20560911505267437, 'ROC AUC': None, 'F1 Score': 0.5553679722333135, 'Time taken': 1.976198434829712}
 11%|████▍                                    | 3/28 [10:40<1:16:03, 182.56s/it]{'Model': 'CalibratedClassifierCV', 'Accuracy': 0.7183181417230227, 'Balanced Accuracy': 0.14424567083056972, 'ROC AUC': None, 'F1 Score': 0.6173181003475434, 'Time taken': 1111.629162788391}
 14%|█████▊                                   | 4/28 [29:12<3:04:30, 461.28s/it]{'Model': 'CheckingClassifier', 'Accuracy': 0.7181221209448202, 'Balanced Accuracy': 0.125, 'ROC AUC': None, 'F1 Score': 0.6003058505604904, 'Time taken': 1.4139881134033203}
 18%|███████▎                                 | 5/28 [29:13<2:03:56, 323.32s/it]{'Model': 'DecisionTreeClassifier', 'Accuracy': 0.8719200235224934, 'Balanced Accuracy': 0.7193825430286243, 'ROC AUC': None, 'F1 Score': 0.8726258496046465, 'Time taken': 74.26043438911438}
 21%|████████▊                                | 6/28 [30:27<1:31:09, 248.60s/it]{'Model': 'DummyClassifier', 'Accuracy': 0.5376653925316084, 'Balanced Accuracy': 0.12623137484057798, 'ROC AUC': None, 'F1 Score': 0.5366471264297887, 'Time taken': 1.3404550552368164}
 25%|██████████▎                              | 7/28 [30:29<1:01:02, 174.42s/it]{'Model': 'ExtraTreeClassifier', 'Accuracy': 0.8722924630010781, 'Balanced Accuracy': 0.7183812439143638, 'ROC AUC': None, 'F1 Score': 0.8727716027076422, 'Time taken': 1.7731153964996338}
 29%|████████████▎                              | 8/28 [30:30<40:52, 122.63s/it]{'Model': 'ExtraTreesClassifier', 'Accuracy': 0.9124179162991277, 'Balanced Accuracy': 0.7124924077775127, 'ROC AUC': None, 'F1 Score': 0.9078155350363198, 'Time taken': 4.871560573577881}
 32%|██████████████▏                             | 9/28 [30:35<27:38, 87.30s/it]{'Model': 'GaussianNB', 'Accuracy': 0.039713809663824366, 'Balanced Accuracy': 0.17552304687363024, 'ROC AUC': None, 'F1 Score': 0.040516361051973, 'Time taken': 2.3686304092407227}
 36%|███████████████▎                           | 10/28 [30:38<18:32, 61.82s/it]{'Model': 'KNeighborsClassifier', 'Accuracy': 0.8006664706458885, 'Balanced Accuracy': 0.44814910725566465, 'ROC AUC': None, 'F1 Score': 0.7875454316170286, 'Time taken': 1057.437790632248}
 46%|███████████████████▌                      | 13/28 [48:17<44:15, 177.06s/it]{'Model': 'LinearDiscriminantAnalysis', 'Accuracy': 0.7116142311084975, 'Balanced Accuracy': 0.22577192174797167, 'ROC AUC': None, 'F1 Score': 0.6407921148776129, 'Time taken': 5.485617637634277}
 50%|█████████████████████                     | 14/28 [48:22<29:18, 125.59s/it]{'Model': 'LinearSVC', 'Accuracy': 0.7156326570616486, 'Balanced Accuracy': 0.17293113469334304, 'ROC AUC': None, 'F1 Score': 0.6201754046599994, 'Time taken': 405.7382245063782}
 54%|██████████████████████▌                   | 15/28 [55:08<45:25, 209.63s/it]{'Model': 'LogisticRegression', 'Accuracy': 0.7178280897775164, 'Balanced Accuracy': 0.1697644256059022, 'ROC AUC': None, 'F1 Score': 0.6301932203169626, 'Time taken': 106.94264650344849}
 57%|████████████████████████                  | 16/28 [56:55<35:45, 178.83s/it]{'Model': 'NearestCentroid', 'Accuracy': 0.5190238165245517, 'Balanced Accuracy': 0.22512012170285664, 'ROC AUC': None, 'F1 Score': 0.5591287831617293, 'Time taken': 1.2205753326416016}
 64%|███████████████████████████▋               | 18/28 [56:57<14:41, 88.15s/it]{'Model': 'PassiveAggressiveClassifier', 'Accuracy': 0.6333627364500637, 'Balanced Accuracy': 0.20506985668049443, 'ROC AUC': None, 'F1 Score': 0.621068415606163, 'Time taken': 5.320954322814941}
 68%|█████████████████████████████▏             | 19/28 [57:02<09:29, 63.30s/it]{'Model': 'Perceptron', 'Accuracy': 0.6349309026756836, 'Balanced Accuracy': 0.16634997421291653, 'ROC AUC': None, 'F1 Score': 0.6063372199874036, 'Time taken': 4.943793773651123}
 71%|██████████████████████████████▋            | 20/28 [57:07<06:06, 45.79s/it]{'Model': 'QuadraticDiscriminantAnalysis', 'Accuracy': 0.3304910320493972, 'Balanced Accuracy': 0.5453811421128513, 'ROC AUC': None, 'F1 Score': 0.3345888578283993, 'Time taken': 5.645366668701172}
 75%|████████████████████████████████▎          | 21/28 [57:13<03:56, 33.75s/it]{'Model': 'RandomForestClassifier', 'Accuracy': 0.906615701264334, 'Balanced Accuracy': 0.6979395245374435, 'ROC AUC': None, 'F1 Score': 0.901478623211132, 'Time taken': 16.19450855255127}
 79%|█████████████████████████████████▊         | 22/28 [57:29<02:50, 28.48s/it]{'Model': 'RidgeClassifier', 'Accuracy': 0.7179064980887974, 'Balanced Accuracy': 0.1342914404881796, 'ROC AUC': None, 'F1 Score': 0.6126863787927609, 'Time taken': 1.4457645416259766}
 82%|███████████████████████████████████▎       | 23/28 [57:31<01:41, 20.37s/it]{'Model': 'RidgeClassifierCV', 'Accuracy': 0.717867293933157, 'Balanced Accuracy': 0.134000437079665, 'ROC AUC': None, 'F1 Score': 0.6122836036775475, 'Time taken': 4.78983211517334}
 86%|████████████████████████████████████▊      | 24/28 [57:35<01:02, 15.70s/it]{'Model': 'SGDClassifier', 'Accuracy': 0.7162011173184357, 'Balanced Accuracy': 0.14794074999431267, 'ROC AUC': None, 'F1 Score': 0.6153410045126061, 'Time taken': 19.367130756378174}
 89%|██████████████████████████████████████▍    | 25/28 [57:55<00:50, 16.80s/it]{'Model': 'SVC', 'Accuracy': 0.7186121728903264, 'Balanced Accuracy': 0.12573602916509266, 'ROC AUC': None, 'F1 Score': 0.6014526860130425, 'Time taken': 3824.947344303131}
 93%|████████████████████████████████████▏  | 26/28 [2:01:40<38:38, 1159.24s/it]{'Model': 'XGBClassifier', 'Accuracy': 0.8765657159658924, 'Balanced Accuracy': 0.6437736527282258, 'ROC AUC': None, 'F1 Score': 0.8646806860345948, 'Time taken': 531.1732261180878}
 96%|██████████████████████████████████████▌ | 27/28 [2:10:31<16:10, 970.82s/it]{'Model': 'LGBMClassifier', 'Accuracy': 0.7989218857198863, 'Balanced Accuracy': 0.4538847969459492, 'ROC AUC': None, 'F1 Score': 0.7549354146645333, 'Time taken': 43.78018379211426}

Extra Trees Classifier is the best: Accuracy': 0.9124179162991277
'''