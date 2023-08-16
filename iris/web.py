from flask import Flask,render_template,request
import pickle
import numpy as np
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/prediction',methods=['POST'])
def predict():
   sepal_length = request.form['SL']
#    print(sepal_length)
   sepal_width = request.form['SW']
#    print(sepal_width)
   petal_length = request.form['PL']
#    print(petal_length)
   petal_width = request.form['PW']
#    print(petal_width)
   li = [sepal_length,sepal_width,petal_length,petal_width]
   li = np.array(li).astype(float).reshape(1,-1)
#    print(li)
   predicted_class = model.predict(li)[0]
   
   return render_template ('result.html',prediction_text="The flower species is {}".format(predicted_class))
if __name__=='__main__':
    app.run(port=8000)

