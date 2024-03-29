# Income prediction model
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

salary_data=pd.read_csv("Salary_Data.csv")
x=salary_data.drop(['Salary'],axis=1)
y=salary_data.drop(['YearsExperience'],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=32)


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
#Fitting the model

regressor=regressor.fit(x_train,y_train)
#Saving the model to disk
pickle.dump(regressor,open('model.pkl','wb') )
#pickle for converting to byte stream, serialising 
#and deserializing

