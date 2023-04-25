import pandas
import numpy
import collinearity
import new_dbscan
import zscore
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import neural_network
from sklearn import metrics
from sklearn import neighbors
from sklearn import tree
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt


#data = pandas.read_csv('minmax_normalised_trackdata.csv')
#data = pandas.read_csv('dupremoved_trackdata.csv')
#data = pandas.read_csv('softmax_normalised_trackdata.csv')
#data = pandas.read_csv('zscore_normalised_trackdata.csv')

# data = pandas.read_csv('outlier_removal/automated_dbscanned_trackdata.csv',index_col=0)
data = pandas.read_csv('outlier_removal/automated_collinearity_removed.csv')

genreMap = sorted(data["genre"].unique())
y = preprocessing.LabelEncoder().fit_transform(data["genre"])
X = data.drop("genre", axis=1)

def neural_net():
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25)
    scaler = StandardScaler()  
    scaler.fit(X_train)  
    X_train = scaler.transform(X_train)  
    X_test = scaler.transform(X_test)
    model = neural_network.MLPClassifier(activation= 'tanh', max_iter= 1000000).fit(X_train, y_train)  


    yPredicted = model.predict(X_test)
    with open('outlier_removal/tests.txt','a') as f:
        accuracy = metrics.accuracy_score(y_test, yPredicted).round(3)
        precision = metrics.precision_score(y_test, yPredicted, average="macro").round(3)
        recall = metrics.recall_score(y_test, yPredicted, average="macro").round(3)
        f.write("\nModel: Neural Net")
        f.write("\nAccuracy: " + str(accuracy))
        f.write("\nPrecision: " + str(precision))
        f.write("\nRecall: " + str(recall) + "\n")
        return accuracy

def SVM():
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25)
    scaler = StandardScaler()  
    scaler.fit(X_train)  
    X_train = scaler.transform(X_train)  
    X_test = scaler.transform(X_test)
    model = svm.SVC(C = 1.5, kernel = 'rbf', degree = 9, gamma = 'scale', decision_function_shape= 'ovo').fit(X_train, y_train)


    yPredicted = model.predict(X_test)
    with open('outlier_removal/tests.txt','a') as f:
        accuracy = metrics.accuracy_score(y_test, yPredicted).round(3)
        precision = metrics.precision_score(y_test, yPredicted, average="macro").round(3)
        recall = metrics.recall_score(y_test, yPredicted, average="macro").round(3)
        f.write("\nModel: SVM")
        f.write("\nAccuracy: " + str(accuracy))
        f.write("\nPrecision: " + str(precision))
        f.write("\nRecall: " + str(recall) + "\n")
        return accuracy
    
def decision_tree():
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25)
    scaler = StandardScaler()  
    scaler.fit(X_train)  
    X_train = scaler.transform(X_train)  
    X_test = scaler.transform(X_test)
    model = tree.DecisionTreeClassifier(criterion = 'entropy', splitter = 'best').fit(X_train, y_train)


    yPredicted = model.predict(X_test)
    with open('outlier_removal/tests.txt','a') as f:
        accuracy = metrics.accuracy_score(y_test, yPredicted).round(3)
        precision = metrics.precision_score(y_test, yPredicted, average="macro").round(3)
        recall = metrics.recall_score(y_test, yPredicted, average="macro").round(3)
        f.write("\nModel: Decision Tree")
        f.write("\nAccuracy: " + str(accuracy))
        f.write("\nPrecision: " + str(precision))
        f.write("\nRecall: " + str(recall) + "\n")
        return accuracy

#dupremoved -> dbscan -> zscore -> collinearity -> k nearest neighbors
best_accuracy = 0
best_min_points = 0
best_eps = 0
methods = ["euclidean", "cosine", "cityblock", "l1", "l2", "hamming"]
best_method = None
prev_removed_songs = 0
best_removed_songs = 0
threshold = .6
for method in methods:
    for min_points in range(1,10):
        prev_removed_songs = 0
        for eps in range(1,100):
            removed_songs = new_dbscan.dbscan_main(eps=eps, min_points=min_points,metric=method)
            if removed_songs == 0:
                with open('outlier_removal/tests.txt','a') as f:
                    f.write("\nRemoved Songs is 0, moving on...\n")
                break
            if removed_songs == prev_removed_songs:
                with open('outlier_removal/tests.txt','a') as f:
                    f.write("\nRemoved Songs is the Same, moving on...\n")
                break
            prev_removed_songs = removed_songs
            zscore.zscore_main()
            collinearity.remove_collinearity(threshold=threshold)
            # accuracy = SVM()
            accuracy = decision_tree()
            if best_accuracy < accuracy:
                best_accuracy = accuracy
                best_min_points = min_points
                best_eps = eps
                best_method = method
                best_removed_songs = prev_removed_songs
with open('outlier_removal/tests.txt','a') as f:
    f.write("\nBest Accuracy: " + str(best_accuracy))
    f.write("\nBest Min_points: " + str(best_min_points))
    f.write("\nBest Epsilon: " + str(best_eps))
    f.write("\nBest Metric: " + str(best_method))
    f.write("\nRemoved Songs: " + str(best_removed_songs))