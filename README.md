# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values. 
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Kishore B
RegisterNumber:  23013723
*/

import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:

### Placement Data:
![image](https://github.com/codedbykishore/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147139122/b6ed8167-9c83-4b31-9e61-75a263cdd213)


### Salary Data:
![image](https://github.com/codedbykishore/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147139122/cc255eb9-3525-4c63-964a-ae03e830e24a)

### Checking the null() function:
![image](https://github.com/codedbykishore/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147139122/b2424db8-9f78-442a-afb4-58765e90fb84)

### Data Duplicate:
![image](https://github.com/codedbykishore/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147139122/884653bb-c2ab-4a7a-9a61-ac27cd264501)

### Print Data:
![image](https://github.com/codedbykishore/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147139122/8de73736-68d1-4760-b4b8-171f191138fd)

### Data-Status:
![image](https://github.com/codedbykishore/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147139122/39afec1d-da89-4eca-977f-a25a05ffbfc9)

### Y_prediction array:
![image](https://github.com/codedbykishore/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147139122/64bd5336-2dfc-43be-bca7-cddd97b446e6)

### Accuracy value:
![image](https://github.com/codedbykishore/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147139122/fd79460f-e6ba-41c0-a0a1-efa7d4eae7f4)

### Classification array:
![image](https://github.com/codedbykishore/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147139122/5a8f4aa9-4615-4c8c-b880-9791169437b9)

### Prediction of LR:
![image](https://github.com/codedbykishore/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147139122/b43bc755-c4ba-4c81-84cf-624f8bd9d4e3)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
