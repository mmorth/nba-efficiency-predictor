# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Ignore benign numpy warnings
import warnings
warnings.filterwarnings("ignore")

# Load NBA player stat dataset
path = "./NBAStats2017Full.csv"
names = ['Age','G','GS','MP','FG','FGA','3P','3PA','2P','2PA','FT','FTA','ORB','DRB','TRB','AST','STL','BLK','TOV','PF','PTS','EFF']
dataset = pandas.read_csv(path, names=names)

# Load dataset
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# dataset = pandas.read_csv(url, names=names)

# Determine how many rows and columns there are for the data
print(dataset.shape)

# Show the first 20 player stat entries
print(dataset.head(20))

# Describe the statistical information about the data
print(dataset.describe())

# Distribution of efficient and not efficient players
print(dataset.groupby('EFF').size())

# Create a box and whisker plots
# dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# plt.show()

# Create histograms of the stats
# dataset.hist()
# plt.show()

# Split-out validation dataset
array = dataset.values
X = array[:,0:21]
Y = array[:,21]
Y=Y.astype('string')
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Check the results of each algorithm
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# Evaluate each model
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Plot algorithms to test performance
# fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.show()

# Make predictions on validation dataset
lr = LogisticRegression()
lr.fit(X_train, Y_train)
predictions = lr.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# Predict an input player
print("Predict a single player: ")
playerStats = [[22,80,80,2845,8.3,15.9,0.6,2.3,7.7,13.7,6,7.7,1.8,7.1,8.9,5.5,1.7,1.9,3,3.1,23.2]]
predict = lr.predict(playerStats)
print(predict)