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

def SVM(C, kernel, degree, break_ties, suppress_output=False):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25)
    scaler = StandardScaler()  
    scaler.fit(X_train)  
    X_train = scaler.transform(X_train)  
    X_test = scaler.transform(X_test)
    if break_ties is False:
        model = svm.SVC(C = C, kernel = kernel, degree = degree, gamma = 'scale', decision_function_shape= 'ovo').fit(X_train, y_train)
    else:
        model = svm.SVC(C = C, kernel = kernel, degree = degree, gamma = 'scale', decision_function_shape= 'ovr', break_ties=break_ties).fit(X_train, y_train)


    yPredicted = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, yPredicted).round(3)
    precision = metrics.precision_score(y_test, yPredicted, average="macro").round(3)
    recall = metrics.recall_score(y_test, yPredicted, average="macro").round(3)
    if suppress_output is False:
        with open('outlier_removal/tests.txt','a') as f:
            f.write("\nModel: SVM")
            f.write("\nAccuracy: " + str(accuracy))
            f.write("\nPrecision: " + str(precision))
            f.write("\nRecall: " + str(recall))
            f.write("\nC: " + str(C))
            f.write("\nKernel: " + str(kernel))
            f.write("\nDegree: " + str(degree))
            f.write("\nBreak Ties: " + str(break_ties) + "\n")
            return accuracy, precision, recall
    return accuracy, precision, recall
    
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

def dbscan_optimization():
    #dupremoved -> dbscan -> zscore -> collinearity -> k nearest neighbors
    best_accuracy = 0
    best_min_points = 0
    best_eps = 0
    methods = ["euclidean", "cosine", "cityblock", "l1", "l2", "hamming"]
    best_method = None
    prev_removed_songs = 0
    best_removed_songs = 0
    threshold = .8
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
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
                total_accuracy, total_precision, total_recall = 0, 0, 0
                for i in range(1,11):
                    accuracy, precision, recall = SVM(C=2.3, kernel='rbf', degree=3, break_ties=False, suppress_output=True)
                    total_accuracy += accuracy
                    total_precision += precision
                    total_recall += recall
                accuracy, precision, recall = total_accuracy/10.0, total_precision/10.0, total_recall/10.0
                with open('outlier_removal/tests.txt','a') as f:
                    f.write("\nAccuracy: " + str(accuracy))
                    f.write("\nPrecision: " + str(precision))
                    f.write("\nRecall: " + str(recall) + "\n")
                # accuracy = decision_tree()
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
        
def hyperparameter_optimization():
    best_accuracy = 0
    best_kernel = 0
    best_break_ties = None
    best_degree = 0
    best_C = 0.0
    kernels = ['poly', 'rbf', 'sigmoid']
    break_tie_list = [True, False]
    for break_ties in break_tie_list:
        for C in range(1, 25):
            for degree in range(1, 10):
                for kernel in kernels:
                    new_dbscan.dbscan_main(eps=14, min_points=5, metric='euclidean')
                    zscore.zscore_main()
                    collinearity.remove_collinearity(threshold=0.8)
                    accuracy = SVM(C=C*0.1,kernel=kernel, degree=degree,break_ties=break_ties)
                    if best_accuracy < accuracy:
                        best_accuracy = accuracy
                        best_kernel = kernel
                        best_break_ties = break_ties
                        best_degree = degree
                        best_C = C
    with open('outlier_removal/tests.txt','a') as f:
        f.write("\nBest Accuracy: " + str(best_accuracy))
        f.write("\nBest Kernel: " + str(best_kernel))
        f.write("\nBest Epsilon: " + str(best_break_ties))
        f.write("\nBest Degree: " + str(best_degree))
        f.write("\nBest C: " + str(best_C))
        
def dbscan_eps_graph():
    #dupremoved -> dbscan -> zscore -> collinearity -> k nearest neighbors
    best_accuracy = 0
    best_eps = 0
    prev_removed_songs = 0
    best_removed_songs = 0
    threshold = .8
    accuracy_list = []
    eps_list = []
    prev_removed_songs = 0
    total_accuracy = 0
    for eps in range(1,100):
        removed_songs = new_dbscan.dbscan_main(eps=eps, min_points=5,metric='euclidean')
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
        total_accuracy = 0
        for i in range(1,11):
            accuracy = SVM(C=2.3, kernel='rbf', degree=3, break_ties=False)
            total_accuracy += accuracy
        accuracy = total_accuracy/50.0
        # accuracy = decision_tree()
        eps_list.append(eps)
        accuracy_list.append(accuracy)
        if best_accuracy < accuracy:
            best_accuracy = accuracy
            best_eps = eps
            best_removed_songs = prev_removed_songs
    with open('outlier_removal/tests.txt','a') as f:
        f.write("\nBest Accuracy: " + str(best_accuracy))
        f.write("\nBest Epsilon: " + str(best_eps))
        f.write("\nRemoved Songs: " + str(best_removed_songs))
    plt.plot(eps_list, accuracy_list)
    plt.title('$\epsilon$\'s Effect on Accuracy')
    plt.xlabel("$\epsilon$")
    plt.ylabel("Accuracy")
    plt.show()

def dbscan_min_pts_graph():
    #dupremoved -> dbscan -> zscore -> collinearity -> k nearest neighbors
    best_accuracy = 0
    best_min_points = 0
    best_method = None
    prev_removed_songs = 0
    best_removed_songs = 0
    threshold = .8
    total_accuracy = 0
    accuracy_list = []
    min_pts_list = []
    for min_points in range(1,10):
        prev_removed_songs = 0
        removed_songs = new_dbscan.dbscan_main(eps=9, min_points=min_points,metric='euclidean')
        prev_removed_songs = removed_songs
        zscore.zscore_main()
        collinearity.remove_collinearity(threshold=threshold)
        total_accuracy = 0
        for i in range(1,11):
            accuracy = SVM(C=2.3, kernel='rbf', degree=3, break_ties=False, suppress_output=True)
            total_accuracy += accuracy
        accuracy = total_accuracy/10.0
        # accuracy = decision_tree()
        min_pts_list.append(min_points)
        accuracy_list.append(accuracy)
        if best_accuracy < accuracy:
            best_accuracy = accuracy
            best_min_points = min_points
            best_removed_songs = prev_removed_songs
    with open('outlier_removal/tests.txt','a') as f:
        f.write("\nBest Accuracy: " + str(best_accuracy))
        f.write("\nBest Min_points: " + str(best_min_points))
        f.write("\nBest Metric: " + str(best_method))
        f.write("\nRemoved Songs: " + str(best_removed_songs))
    plt.plot(min_pts_list, accuracy_list)
    plt.title('Min Points Effect on Accuracy')
    plt.xlabel("Min Points")
    plt.ylabel("Accuracy")
    plt.show()

dbscan_optimization()
# hyperparameter_optimization()
# dbscan_eps_graph()
# dbscan_min_pts_graph()