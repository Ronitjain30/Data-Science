import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("\n")


'''         Looking and visualizing the data :-          '''

data = pd.read_csv("D:\Internship\Week 5\Wine.csv", 
                    usecols=['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 
                                'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 
                                    'OD280/OD315 of diluted wines', 'Proline', 'Class'])

print("DATA :-\n", data.head(), "\n")

#Checking for null values :-
print("NULL OR MISSING VALUES :-")
print(data.isnull().sum(), "\n")

#Statistics :-
print("DATA STATISTICS :-", "\n",data.describe(), "\n")

#Columns :-
print("COLUMNS :-\n", data.columns, "\n")
print("NO. OF COLUMNS :- ", len(data.columns), "\n")



'''      Separating Target Variable and Splitting Dataset:-     '''

X = data.drop(["Class"], axis=1)
y = data["Class"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=1)



'''      Confusion Matrix :-     '''

import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.BuGn):

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    # plt.show()



'''      Traning With Entropy (Decision Tree):-     '''

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

classifier = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)
classifier = classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print("Accuracy : ", metrics.accuracy_score(y_test, y_pred), "\n")


confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

plt.figure()
print("CONFUSION MATRIX :-")
plot_confusion_matrix(confusion_matrix, classes=['1', '2', '3'], title='Decision Tree')


# #Predictions :-
# print("Predictions :", y_pred, "\n")