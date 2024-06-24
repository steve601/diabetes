from flask import Flask,request,render_template
import pickle
import numpy as np  

app = Flask(__name__)

def load_model():
    with open('diabetes.pkl','rb') as file:
        data = pickle.load(file)
    return data

objects = load_model()
model = objects['model']
scaler = objects['scaler']

@app.route('/')
def homepage():
    return render_template('diabetes.html')

@app.route('/submit',methods = ['POST'])
def predict():
    bmi = request.form.get('bmi')
    degree = request.form.get('diabetespedigreefunction')
    age = request.form.get('age')
    
    x = np.array([[bmi,degree,age]])
    x = scaler.transform(x)
    predicted = model.predict(x)
    
    msg = 'Patient has diabetes' if predicted == 1 else 'Patient has no diabetes'
    
    return render_template('diabetes.html',text = msg)

if __name__ == '__main__':
    app.run(host="0.0.0.0")
