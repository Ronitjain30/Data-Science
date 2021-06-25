import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("\n")


'''         Looking and visualizing the data :-          '''

data = pd.read_excel("D:\Internship\Week 5\Heart_failure.xlsx")
print("DATA :-\n", data.head(), "\n")

#Checking for null values :-
print("NULL OR MISSING VALUES :-")
print(data.isnull().sum(), "\n")

#Statistics :-
print("DATA STATISTICS :-", "\n",data.describe(), "\n")

#Columns :-
print("COLUMNS :-\n", data.columns, "\n")
print("NO. OF COLUMNS :- ", len(data.columns), "\n")



'''       Transforming the data :-       '''

#Converting object type data to numerical data :-
Categorical_data = []
Numerical_data = []

for i,j in enumerate(data.dtypes):
    if j == object:
        Categorical_data.append(data.iloc[:,i])
    else:
        Numerical_data.append(data.iloc[:,i])

Categorical_data = pd.DataFrame(Categorical_data).transpose()
Numerical_data = pd.DataFrame(Numerical_data).transpose()

# print(Categorical_data.head())
# print(Numerical_data.head())

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

for i in Categorical_data:
    Categorical_data[i] = LE.fit_transform(Categorical_data[i])
# print(Categorical_data.head())

data_updated = pd.concat([Numerical_data, Categorical_data], axis=1)
print("DATA TRANSFORMED :-\n", data_updated.head(),"\n")



'''      Separating Target Variable :-     '''

X = data_updated.drop(["DEATH_EVENT"], axis=1)
y = data_updated["DEATH_EVENT"]



'''      Feature Selection :-       '''

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

bestfeatures = SelectKBest(score_func=chi2, k=12)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Factors','Score']                #naming the dataframe columns
print("FEATURE SELECTION :-")
print(featureScores.nlargest(12,'Score'))

print("Therefor we will select Top 6 factors which are : platelets, time, creatinine_phosphokinase, ejection_fraction, age and serum_creatinine.\n")



'''       Splitting the Data :-      '''

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X[['platelets', 'time', 'creatinine_phosphokinase', 'ejection_fraction', 'age',  'serum_creatinine']], 
                                                        y, test_size=0.2, random_state=0)



'''     Feature Scaling Normalization :-     '''

from sklearn import preprocessing
scaler_ss = preprocessing.StandardScaler()
X_train_scaled = scaler_ss.fit_transform(X_train)
X_test_scaled = scaler_ss.fit_transform(X_test)



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



'''       SUPPORT VECTOR MACHINE :-      '''

from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.svm import SVC

SVC_param = {'kernel':['sigmoid','rbf','poly', 'linear'], 'C':[1], 'decision_function_shape':['ovr'], 'random_state':[0]}
SVC_optimal_param = GridSearchCV(SVC(), SVC_param, cv=None)
SVC_optimal_param.fit(X_train_scaled, y_train)

y_pred = SVC_optimal_param.predict(X_test_scaled)

print("The best parameters are :- ",SVC_optimal_param.best_params_, "\n")

print("Accuracy :-  {0:.3f}".format(metrics.accuracy_score(y_test, y_pred)), "\n")

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

plt.figure()
print("CONFUSION MATRIX :-")
plot_confusion_matrix(confusion_matrix, classes=['Alive', 'Death'], title='SVM')