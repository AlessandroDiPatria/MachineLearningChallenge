
# Online news popularity;
![Schermata 2022-12-07 alle 23 34 37](https://user-images.githubusercontent.com/82099379/206311522-4e84680a-a3c8-4d4a-bd6e-3d4bc08503f6.png)

The provided dataset consists of one single csv file ("OnlineNewsPopularity.csv");
The provided dataset is a modified noisy version of the original dataset described in [1];
[1] K. Fernandes, P. Vinagre and P. Cortez. A Proactive Intelligent Decision Support System for Predicting the Popularity of Online News. Proceedings of the 17th EPIA 2015 - Portuguese Conference on Artificial Intelligence, September, Coimbra, Portugal

This dataset summarizes a heterogeneous set of features about articles published by Mashable in a period of two years. The goal of the task is to predict the number of shares in social networks (popularity).

Number of Instances: 39,797

Number of Attributes: 61

Target: shares


## Cleaning and Prepocessing

### Remove or impute "null" values
Remove all null values throught simple imputer and check if is correctly applied on our dataset
![Schermata 2023-01-02 alle 12 32 01](https://user-images.githubusercontent.com/82099379/210225738-396290a2-c900-4705-b129-5a22848f912f.png)


### Scaling
Scaling the Dataset with MinMax Scaling
![Schermata 2023-01-02 alle 12 31 00](https://user-images.githubusercontent.com/82099379/210225620-e65e3e17-2410-4fd2-a741-50e2dfe34e2f.png)

### Discretize share columns
![Schermata 2023-01-02 alle 12 36 56](https://user-images.githubusercontent.com/82099379/210226211-8a1ceed7-885d-4548-a1b8-6064fe431746.png)

### Re-Sampling  using SMOTE
Re-Sampling share columns using SMOTE
![Schermata 2023-01-02 alle 12 38 02](https://user-images.githubusercontent.com/82099379/210226321-33b60622-5b26-4737-93f9-c4142566271a.png)






## Dataset Analysis

1. Total Number of Samples:
39648

2. Table with 15 examples

3. Plot no-discretize share columns

![Schermata 2023-01-02 alle 12 33 37](https://user-images.githubusercontent.com/82099379/210225907-c1f006c8-778b-4d8d-8133-e7f4132e0100.png)

4. A bar chart counting the attributes: data_channel_is_lifestyle, data_channel_is_entertainment, data_channel_is_bus, data_channel_is_socmed, data_channel_is_tech, data_channel_is_world;

![Schermata 2023-01-02 alle 12 36 22](https://user-images.githubusercontent.com/82099379/210226172-61dab5a0-fc87-4fc6-b04b-fb4943b1dd4e.png)


##Feature importance analysis 
Perform feature importance analysis
![Schermata 2023-01-02 alle 12 39 08](https://user-images.githubusercontent.com/82099379/210226416-d7a7f4e0-56d7-40b0-b26b-ad9b526ffbe2.png)

*Filter ony useful features*

![Schermata 2023-01-02 alle 12 39 33](https://user-images.githubusercontent.com/82099379/210226453-7f4be615-7f14-41c8-b477-532194bece9d.png)


## Model Selection 

### Decision Three 

![Schermata 2023-01-02 alle 12 41 17](https://user-images.githubusercontent.com/82099379/210226643-e20f7ae7-cbd6-494b-947a-72dee1254383.png)

### SVM 
![Schermata 2023-01-02 alle 12 41 41](https://user-images.githubusercontent.com/82099379/210226673-8e8f5a76-e71f-4339-84c8-b7131d9e545b.png)


### Ensamble Methods 
![Schermata 2023-01-02 alle 12 42 12](https://user-images.githubusercontent.com/82099379/210226724-56d19016-6d00-47e0-8892-1afc84974d4a.png)


### Multilayer Perceptron Network
![Schermata 2023-01-02 alle 12 42 40](https://user-images.githubusercontent.com/82099379/210226771-9f069990-588f-4107-88fb-0044fd25c785.png)




## Resume 

In the first part I tried plotting some statistics about the dataset as the number of null/none values, general statistics using "data.describe" that gives us min, max,std count and many others for each feature. The null values were very few. It could be easily deleted but I opted to replace them. I decided to replace all none and null values using Simple Imputer with the strategy mean. In order to increase accuracy of each classifier I decided to scale the dataset between 0,1 with MinMax classifier.

In the second Part after plotting the numbers of columns, the first 15 columns of table, the distribution of share and the A bar chart counting the attributes required I decide to discretize with "K-bins Discretizer '' the share column in 6 bins using quantile strategy. Discretize means divide the continuous values of share in 6 different classes called bins. I noticed in the plot that bins are unbalanced so I decided to resample it. I used the SMOTE algorithm technique, which creates new, artificial samples by tracing lines between pre-existing samples in their feature space and choosing points in these lines to be added to the dataset. As you can see in the relative plot, after SMOTE our classes are perfectly balanced. After I decide to find the most relevant features in data in order to construct a solid and accurate model. I decided to use a "Random Forest" model and "feature_importances" that gives us the features most relevant to predict the y. I decided to create a new dataset with the only selected features. Finally I plot the result of feature importance analysis highlighting that I found about 25 relevant features from the 60 originally.

In the third part I create all the models required to predict “share” columns.I decide to make crossValidation for each model and use GridParams to tune Hyperparameters. For the evaluation phase I decided to use a Cross Validation and classification report that gives the results in terms of accuracy, f1 score, precision and recall. The F1-score combines the precision and recall of a classifier into a single metric by taking their harmonic mean. I decided to use it in order to compare the performance of classifiers.

I started with Decision Tree, I applied cross validation with 10 folds and hyper parameters tuning with the following parameters to be tuned "('max_leaf_nodes': list(range(2, 20)), 'min_samples_split': [2, 3, 4, 5])" The best parameters fund is "(max_leaf_nodes=15, random_state=42" The results show an accuracy around 22%. Decision Tree is one of the oldest ML models and its accuracy is low with respect new and more complicated models like Neural Network.

After I create a SVM model, I applied cross validation with 2 folds and hyper parameter tuning with the following parameters to be tuned " 'C': [0.1, 1],'gamma': [0.1, 0.01 ],'kernel': ['rbf']" " The best parameters fund is "(max_leaf_nodes=15, random_state=42" The results shows an accuracy around the "25%"

I create models using Ensemble methods combining LogisticRegression, GaussianNaiveBayes ,RandomForestClassifier, VotingClassifier. I applied cross validation with 3 folds and I tuned the following Hyperparameter : " 'lr__C': [1.0, 100.0], 'rf__n_estimators': [20, 200] " The best parameters found are "'lr__C': 100.0, 'rf__n_estimators': 20". The results show an accuracy around the "20%".

Finally the last model created is Multilayer Perceptron Network. As usually i tuned some hyperparameters : "epochs=[10,20,30]" The best parameters found is "MLPClassifier(activation='tanh', hidden_layer_sizes=(10, 30, 10),random_state=1)" The results shows an accuracy around the "30%" In general all the models trained perform very similarly with an accuracy of 30%. MLP seems to perform better because NN are generally faster but at the same time complex respect other methods.
