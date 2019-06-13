from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import pandas as pd
import data_loader
def randomSearch(processed_train_features,processed_valid_features,train_labels,valid_labels,processed_test_features,test_labels):
	n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
	n_estimators= [10]
	max_features = ['auto', 'sqrt']
	max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
	max_depth = [3]
	max_depth.append(None)
	min_samples_split = [2, 5, 10]
	min_samples_leaf = [1, 2, 4]
	bootstrap = [True, False]
	# Create the random grid
	random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
	print(random_grid)
	rf = RandomForestRegressor()
	rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=2, random_state=42, n_jobs = -1)


	rf_random.fit(processed_train_features, train_labels)
	train_predict = rf_random.predict(processed_train_features)

	errors = abs(train_predict - train_labels)
	mape = 100 * np.mean(errors)
	print(mape)
	accuracy = 100 - mape
	print("Accuracy ",accuracy)


	rf_random.fit(processed_valid_features, valid_labels)
	valid_predict = rf_random.predict(processed_valid_features)
	errors = abs(valid_predict - valid_labels)
	mape = 100 * np.mean(errors)
	#print(mape)
	accuracy = 100 - mape
	print("Accuracy ",accuracy)
	test_predict = rf_random.predict(processed_test_features)
	errors = abs(test_predict - test_labels)
	mape = 100 * np.mean(errors)
	accuracy = 100 - mape
	print("Accuracy on Test Data",accuracy)
