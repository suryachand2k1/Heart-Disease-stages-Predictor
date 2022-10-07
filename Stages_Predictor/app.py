from flask import Flask, render_template, url_for, request
import joblib
import os
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__, static_folder='static')

dataset = pd.read_csv("recons_dataset/combined_dataset.csv")

from sklearn.model_selection import train_test_split

predictors = dataset.drop("num",axis=1)
target = dataset["num"]

X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(predictors, target)
predictions = clf.predict(X_test)

@app.route("/")
def index():
    return render_template('home.html')


@app.route('/result', methods=['POST', 'GET'])
def result():
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    trestbps = float(request.form['trestbps'])
    chol = float(request.form['chol'])
    restecg = float(request.form['restecg'])
    thalach = float(request.form['thalach'])
    exang = int(request.form['exang'])
    cp = int(request.form['cp'])
    fbs = float(request.form['fbs'])
    x = np.array([age, sex, cp, trestbps, chol, fbs, restecg,
                  thalach, exang]).reshape(1, -1)


    y = clf.predict(x)
    print(y)

    # No heart disease
    if y == 0:
        return render_template('nodisease.html')

    # y=1,2,4,4 are stages of heart disease
    else:
        return render_template('heartdisease.htm', stage=int(y))


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == "__main__":
    app.run(debug=True)
