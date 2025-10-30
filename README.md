# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the libraries and read the data frame using pandas.

2.Calculate the null values present in the dataset and apply label encoder.

3.Determine test and training data set and apply decison tree regression in dataset.

4.Calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Dharshini.S
RegisterNumber:212224230061
*/
```
```

import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])

```

## Output:

# Data Head:

<img width="390" height="265" alt="image" src="https://github.com/user-attachments/assets/67d5163b-2084-4ff1-a6da-c92455b576ee" />

# Data Info:

<img width="603" height="237" alt="image" src="https://github.com/user-attachments/assets/3838d84f-9d0b-4065-ae67-803bc3dc3cda" />

# isnull() sum():

<img width="603" height="237" alt="image" src="https://github.com/user-attachments/assets/d5733fda-4473-49fe-b2a2-6505a1e97371" />

# Data Head for salary:

<img width="323" height="234" alt="image" src="https://github.com/user-attachments/assets/25a2b015-0304-4745-a603-029c76d9ec26" />

# Mean Squared Error :

<img width="239" height="38" alt="image" src="https://github.com/user-attachments/assets/f047515b-622b-425c-b4e9-078976c7e760" />

# r2 Value:

<img width="1065" height="41" alt="image" src="https://github.com/user-attachments/assets/3c2635bf-97da-49d8-a484-cc66e1c40633" />

# Data prediction :

Data prediction :<img width="311" height="38" alt="image" src="https://github.com/user-attachments/assets/68e3656d-d7a7-41dc-abad-1edfe5dd7c9b" />

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
