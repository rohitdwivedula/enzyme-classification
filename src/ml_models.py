# Base Libraries
import argparse
import csv
import sys
import pickle
import numpy as np
import os
import pandas as pd
import time

# The Models
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost
import lightgbm

'''
We perform Stratified K fold cross validation coupled with various sampling strategies and 
store the confusion matrix and the classification report.
''' 
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
SAMPLING_METHODS = ["NONE", "SMOTE"]

parser = argparse.ArgumentParser(description='Run all ML models on protein dataset.')
parser.add_argument('--smoketest', dest='smoketest', action='store_true', help='Run models on only first 100 rows of data (for testing)')
parser.set_defaults(smoketest=False)
parser.add_argument('--expensive_classifier', dest='expensive_classifier', action='store_true', help='Run models on time taking classifiers only')
parser.set_defaults(expensive_classifier=False)
args = parser.parse_args()

if not args.expensive_classifier:
	ALL_MODELS = [PassiveAggressiveClassifier, Perceptron, RidgeClassifier, SGDClassifier, 
			LogisticRegression, LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis,
			BernoulliNB, GaussianNB, KNeighborsClassifier, NearestCentroid, 
			DecisionTreeClassifier, AdaBoostClassifier, lightgbm.LGBMClassifier]

	ALL_MODEL_NAMES = ['PassiveAggressiveClassifier', 'Perceptron', 'RidgeClassifier', 'SGDClassifier', 
			'LogisticRegression', 'LinearDiscriminantAnalysis', 'QuadraticDiscriminantAnalysis',
			'BernoulliNB', 'GaussianNB', 'KNeighborsClassifier', 'NearestCentroid', 
			'DecisionTreeClassifier', 'AdaBoostClassifier', 'LGBM']
			
else:
	ALL_MODELS = ["SVC", "LinearSVC", "xgboost"]
	ALL_MODEL_NAMES = [SVC, LinearSVC, xgboost.XGBClassifier]

with open('../data/X.pickle','rb') as infile:
	X = pickle.load(infile)

with open('../data/y.pickle','rb') as infile:
	y = pickle.load(infile)

X = np.reshape(X, newshape=(127537, 300))

# removing translocales
X = X[y != 7]
y = y[y != 7]

if args.smoketest:
	X = X[:100]
	y = y[:100]

kf = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

if os.path.exists('results.pickle'):
	with open('results.pickle', 'rb') as f:
		all_results = pickle.load(f)
	# model': 'LogisticRegression', 'fold': 2, 'sampling': 'SMOTETomek
	resumed_run = True
	last_iter_fold = all_results[-1]['fold']
	last_iter_model = all_results[-1]['model']
	last_iter_sampling = all_results[-1]['sampling']
else:
	all_results = []
	resumed_run = False
	last_iter_fold = None
	last_iter_model = None
	last_iter_sampling = None

fold = 1
for train_index, test_index in kf.split(X, y):
	if resumed_run:
		if fold < last_iter_fold:
			# this fold has already been done, skip
			print("K Fold Cross Validation || Fold #", fold, "already done. Skipped.")
			fold+=1
			continue
	print("K Fold Cross Validation || Fold #", fold)
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	
	for sampling_method in SAMPLING_METHODS:
		print("K Fold", fold, "sampling methods begin")
		if resumed_run:
			if SAMPLING_METHODS.index(last_iter_sampling) > SAMPLING_METHODS.index(sampling_method):
				# this sampling method has already been done, skip
				print("Fold #", fold, ", sampling", sampling_method, "already done. Skipped.")
				continue

		print("Sampling strategy", sampling_method, "begun.")
		start = time.time()
		if sampling_method == "NONE":
			X_resampled, y_resampled = X_train, y_train
		
		elif sampling_method == "SMOTE":
			X_resampled, y_resampled = SMOTE(random_state=1).fit_resample(X_train, y_train)
		
		# elif sampling_method == "ADASYN":
		# 	X_resampled, y_resampled = ADASYN(random_state=1).fit_resample(X_train, y_train)
		
		# elif sampling_method == "SMOTEENN":
		# 	print("SMOTEENN skipped")
		# 	continue
		# 	smote_enn = SMOTEENN(random_state=1)
		# 	X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
		
		# elif sampling_method == "SMOTETomek":
		# 	print("SMOTETomek skipped")
		# 	resumed_run = False
		# 	continue
		# 	smote_tomek = SMOTETomek(random_state=1)			
		# 	X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
		
		print("Sampling of", sampling_method, "done. Took %.2f"%(time.time()-start))
		for (classifier, model_name) in zip(ALL_MODELS, ALL_MODEL_NAMES):
			if resumed_run:
				if ALL_MODEL_NAMES.index(last_iter_model) > ALL_MODEL_NAMES.index(model_name):
					# this sampling method has already been done, skip
					print("Fold #", fold, ", sampling", sampling_method, "classifier", model_name, "already done. Skipped.")
					continue
				elif ALL_MODEL_NAMES.index(last_iter_model) == ALL_MODEL_NAMES.index(model_name):
					print("Fold #", fold, ", sampling", sampling_method, "classifier", model_name, "already done. Skipped.")
					resumed_run = False
					continue
			print("Running on model: ", model_name, "with", sampling_method, "sampling method on Fold #", fold)
			clf = classifier()
			start_train = time.time()
			clf.fit(X_resampled, y_resampled)
			end_train = time.time()
			y_pred = clf.predict(X_test)
			end_test = time.time()
			results_dict = {
				"model": model_name,
				"fold": fold,
				"sampling": sampling_method, 
				"confusion_matrix": confusion_matrix(y_test, y_pred),
				"report": classification_report(y_test, y_pred, output_dict = True),
				"train_time": end_train - start_train,
				"test_time": end_test - end_train
			}
			all_results.append(results_dict)
			with open('results.pickle', 'wb') as handle:
				pickle.dump(all_results, handle)
			print("Model", model_name, "on fold", fold, "with sampling strategy", sampling_method, "completed in total of", time.time()-start_train, "seconds")
	fold += 1

print("All runs complete.")