
# News Popularity Data Analysis

In the first part I tried plotting some statistics about the dataset as the number of null/none values, general statistics using "data.describe" that gives us min, max,std count and many others for each feature. The null values were very few. It could be easily deleted but I opted to replace them. I decided to replace all none and null values using Simple Imputer with the strategy mean. In order to increase accuracy of each classifier I decided to scale the dataset between 0,1 with MinMax classifier.

In the second Part after plotting the numbers of columns, the first 15 columns of table, the distribution of share and the A bar chart counting the attributes required I decide to discretize with "K-bins Discretizer '' the share column in 6 bins using quantile strategy. Discretize means divide the continuous values of share in 6 different classes called bins. I noticed in the plot that bins are unbalanced so I decided to resample it. I used the SMOTE algorithm technique, which creates new, artificial samples by tracing lines between pre-existing samples in their feature space and choosing points in these lines to be added to the dataset. As you can see in the relative plot, after SMOTE our classes are perfectly balanced. After I decide to find the most relevant features in data in order to construct a solid and accurate model. I decided to use a "Random Forest" model and "feature_importances" that gives us the features most relevant to predict the y. I decided to create a new dataset with the only selected features. Finally I plot the result of feature importance analysis highlighting that I found about 25 relevant features from the 60 originally.

In the third part I create all the models required to predict “share” columns.I decide to make crossValidation for each model and use GridParams to tune Hyperparameters. For the evaluation phase I decided to use a Cross Validation and classification report that gives the results in terms of accuracy, f1 score, precision and recall. The F1-score combines the precision and recall of a classifier into a single metric by taking their harmonic mean. I decided to use it in order to compare the performance of classifiers.

I started with Decision Tree, I applied cross validation with 10 folds and hyper parameters tuning with the following parameters to be tuned "('max_leaf_nodes': list(range(2, 20)), 'min_samples_split': [2, 3, 4, 5])" The best parameters fund is "(max_leaf_nodes=15, random_state=42" The results show an accuracy around 22%. Decision Tree is one of the oldest ML models and its accuracy is low with respect new and more complicated models like Neural Network.

After I create a SVM model, I applied cross validation with 2 folds and hyper parameter tuning with the following parameters to be tuned " 'C': [0.1, 1],'gamma': [0.1, 0.01 ],'kernel': ['rbf']" " The best parameters fund is "(max_leaf_nodes=15, random_state=42" The results shows an accuracy around the "25%"

I create models using Ensemble methods combining LogisticRegression, GaussianNaiveBayes ,RandomForestClassifier, VotingClassifier. I applied cross validation with 3 folds and I tuned the following Hyperparameter : " 'lr__C': [1.0, 100.0], 'rf__n_estimators': [20, 200] " The best parameters found are "'lr__C': 100.0, 'rf__n_estimators': 20". The results show an accuracy around the "20%".

Finally the last model created is Multilayer Perceptron Network. As usually i tuned some hyperparameters : "epochs=[10,20,30]" The best parameters found is "MLPClassifier(activation='tanh', hidden_layer_sizes=(10, 30, 10),random_state=1)" The results shows an accuracy around the "30%" In general all the models trained perform very similarly with an accuracy of 30%. MLP seems to perform better because NN are generally faster but at the same time complex respect other methods.