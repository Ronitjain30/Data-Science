import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



'''           Regression Equation :-         
SP = I + c1*Y + c2*KD + c3*F + c4*ST + c5*T + c6*O

where,
ci's = coefficients
SP = selling_price
Y = year
KD = km_driven
F = fuel
ST = seller_type
T = transmission
O = owner             '''



'''         Looking and visualizing the data :-          '''

data = pd.read_csv("Car_details-1.csv")
print(data.head(), "\n")

#Dropping "car_name" column :-
data.drop("name", axis=1, inplace=True)

#Checking for null values :-
print(data.isnull().sum(), "\n")

#Statistics :-
print(data.describe(), "\n")

#Columns :-
print(data.columns, "\n")

#Plotting some graphs to get some insights :-
# sns.countplot(data["fuel"])
# plt.show()

# sns.countplot(data["seller_type"])
# plt.show()

# sns.countplot(data["transmission"])
# plt.show()

# sns.countplot(data["owner"])
# plt.show()

#Graph for Average selling price comparison :-
# graph = data.groupby(["year", "owner"])["selling_price"].mean().reset_index()
# plt.figure(figsize=(20,10))
# sns.barplot(x="year", y="selling_price",hue="owner",data=graph)
# plt.xlabel("year")
# plt.ylabel("Average Selling Price")
# plt.show()



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
print(data_updated.head(),"\n")



'''       Finding Correlation :-        '''

print(data_updated.corr(), "\n")

#Making Correlation Matrix :-
# plt.figure(figsize=(10,8))
# plt.title = "Correlation Matrix"
# sns.heatmap(data_updated.corr(), annot=True)
# plt.show()



'''      Training the model and Applying Linear Regression:-      '''

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

X = data_updated.drop(["selling_price"], axis=1)
Y = data_updated["selling_price"]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=1)

regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#Intercept and Coefficients :-
Intercept = regressor.intercept_
Coefficient = regressor.coef_
print("Intercept : ",Intercept)
print("Coefficients : ",Coefficient,'\n')



'''      Testing and Evaluating the Model :-      '''

#Example :-
# car = np.array([2007,70000,4,1,1,0])
# y_pred_car = regressor.predict(car.reshape(1,-1))
# print('Predicted Car Price : ', y_pred_car)


y_pred = regressor.predict(X_test)
print('R_squared : ', metrics.r2_score(Y_test, y_pred))
print('Mean Absolute Error : ', metrics.mean_absolute_error(Y_test, y_pred))
print("\n")


# sns.distplot(Y_test-y_pred)
# plt.show()
# plt.scatter(Y_test, y_pred)
# plt.show()


'''     User input to predict Car Price :-    '''

def take_user_input():
    year = int(input('Enter the year : '))
    km_driven = int(input('Enter the km_driven : '))
    fuel = int(input('Enter the type of fuel : '))
    seller_type = int(input('Enter the type of seller_type : '))
    transmission = int(input('Enter the type of transmission : '))
    owner = int(input('Enter the type of owner : '))

    test_car = np.array([year,km_driven,fuel,seller_type,transmission,owner])
    predicted_price = regressor.predict(test_car.reshape(1,-1))
    print('Predicted Price for the car : ',predicted_price)


#Uncomment the following to take user input :-
# take_user_input()
