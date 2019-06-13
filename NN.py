from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import data_loader
def plot2DMat(matrix, X, Y, Xtitle, Ytitle, title):
    #print("hi")
    fig, ax = plt.subplots(1,1, figsize=(10,8))
    #Validation AUC plot
    cax1 = ax.matshow(matrix, interpolation = "nearest")
    fig.colorbar(cax1)
    ax.set_xticklabels([''] + list(X))
    ax.set_yticklabels([''] + list(Y))
    ax.set_title(title)
    ax.legend()
    ax.set(xlabel=Xtitle, ylabel=Ytitle)
    plt.show()
#plot2DMat(rf_train_accuracy_res, , , 'max_depth', 'n_estimator', 'Training Accuracy for random forest')
def NeuralNets(processed_train_features,processed_valid_features,train_labels,valid_labels,processed_test_features,test_labels):
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
	clf.fit(processed_train_features,train_labels)
	y_train = clf.predict(processed_train_features)
	y_valid = clf.predict(processed_valid_features)
	print("Neural Nets Training accuracy ",accuracy_score(train_labels,y_train ))
	print("Neural Nets Validation accuracy ",accuracy_score(valid_labels, y_valid))
	#learning_rate=0.01
	train_list = np.zeros((5, 5))
	valid_list = np.zeros((5, 5))
	test_list = np.zeros((5, 5))
	learning_rate = np.linspace(0.01,0.05,5)
	hidden_layer_sizes = [7,9,11,13,15]
	#learning_rate = [0.1,0.2,0.3]
	for i in range(0,5):
		for j in range(0,5):
			clf1 = MLPClassifier(alpha=learning_rate[i],hidden_layer_sizes=(i+5, j+2),random_state=0)
			clf1.fit(processed_train_features,train_labels)
			y_train = clf1.predict(processed_train_features)
			y_valid = clf1.predict(processed_valid_features)
			y_test = clf1.predict(processed_test_features)
			train_list[i][j] = accuracy_score(train_labels,y_train)
			valid_list[i][j]= accuracy_score(valid_labels, y_valid)
			test_list[i][j]=accuracy_score(test_labels, y_test)
	plot2DMat(train_list*100, hidden_layer_sizes, learning_rate, "Hidden Layer Size", "Learning Rate", "Train Accuracy for neural Net")
	plot2DMat(valid_list*100, hidden_layer_sizes, learning_rate, "Hidden Layer Size", "Learning Rate", "Validation Accuracy for neural Net")
	plot2DMat(test_list*100, hidden_layer_sizes, learning_rate, "Hidden Layer Size", "Learning Rate", "Test Accuracy for neural Net")