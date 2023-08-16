import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
data = pd.read_excel('iris .xls')


x= data.drop('Classification',axis=1)
y=data.drop(['SL','SW','PL','PW'],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()

model.fit(x_train,y_train)

pickle.dump(model,open('model.pkl','wb') )