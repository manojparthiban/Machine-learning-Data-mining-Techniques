# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
import warnings
warnings.filterwarnings("ignore")

# Importing the dataset
dataset = pd.read_csv('orbit classification for prediction.csv')

# print(dataset)
X = dataset.iloc[:, [0,1,2,3,4,5,6,7,8]].values  # features to predict target
y = dataset.iloc[:, -1].values # target 
acc =  []
auc_score = []
all_classifier = ["KNN","NAIVE BAYES","DECISION TREE","SVM"]
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


def performance_of_classifier(classifier,clf,X_test,y_test,y_pred):
    print("---------------PERFORMANCE ANALYSIS FOR {} CLASSIFIER----------------\n".format(clf))

    print("Real Test dataset labels: \n{}\n".format(y_test))
    print("Predicted Test dataset labels: \n{}\n".format(y_pred))

    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(classifier, X_test, y_test,cmap=plt.cm.Blues)  
    plt.show()
    

def auc_roc(classifier,c,clf,X_test,y_test):
    y_test = lb.fit_transform(y_test)
    prob_test = np.squeeze(classifier.predict_proba(X_test)[:,1].reshape(1,-1))
    fpr, tpr, thresholds1 = metrics.roc_curve(y_test, prob_test)
    auc = metrics.auc(fpr, tpr)
    plt.title("AUC - ROC curve")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.plot(fpr, tpr, color = c,label = clf)
    return auc

    
# Training the K-NN model on the Training set
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 4, metric = 'minkowski', p = 2)
knn.fit(X_train, y_train)

# Predicting the Test set results
y_pred = knn.predict(X_test)

performance_of_classifier(knn,"KNN",X_test,y_test,y_pred)
acc.append(int(metrics.accuracy_score(y_test, y_pred)*100))
print("Accuracy of KNN: {}%\n\n".format(acc[-1]))


#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
nbc = GaussianNB()

#Train the model using the training sets
nbc.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = nbc.predict(X_test)

performance_of_classifier(nbc,"NAIVE BAYES",X_test,y_test,y_pred)
acc.append(int(metrics.accuracy_score(y_test, y_pred)*100))
print("Accuracy of  NAIVE BAYES: {}%\n\n".format(acc[-1]))


#Import DecisionTreeClassifier model
from sklearn.tree import DecisionTreeClassifier

#Create a decision tree Classifier
dect = DecisionTreeClassifier() # Linear Kernel
dect.fit(X_train, y_train)

# Predicting the Test set results
y_pred = dect.predict(X_test)

performance_of_classifier(dect,"DECISION TREE",X_test,y_test,y_pred)
acc.append(int(metrics.accuracy_score(y_test, y_pred)*100))
print("Accuracy of  DECISION TREE: {}%\n\n".format(acc[-1]))



#Import svm model
from sklearn import svm

#Create a svm Classifier
svm = svm.SVC(kernel='linear',probability = True) # Linear Kernel
svm.fit(X_train, y_train)

# Predicting the Test set results
y_pred = svm.predict(X_test)

performance_of_classifier(svm,"SVM",X_test,y_test,y_pred)
acc.append(int(metrics.accuracy_score(y_test, y_pred)*100))
print("Accuracy of  SVM: {}%".format(acc[-1]))


auc_score.append(auc_roc(knn,"blue",all_classifier[0],X_test,y_test))
auc_score.append(auc_roc(nbc,"orange",all_classifier[1],X_test,y_test))
auc_score.append(auc_roc(dect,"red",all_classifier[2],X_test,y_test))
auc_score.append(auc_roc(svm,"green",all_classifier[3],X_test,y_test))
plt.legend()
plt.grid()
plt.show()

print("\nAUC_ROC score:\n")
for i in range(len(all_classifier)): print("{} classifier : {}".format(all_classifier[i],round(auc_score[i],1) ) )
    
print("\nFor this dataset , {} classifier is best with accuracy {}% and auc_roc score {}".format(all_classifier[auc_score.index(max(auc_score))],max(acc),round(max(auc_score),1)))
