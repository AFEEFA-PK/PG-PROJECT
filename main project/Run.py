#importing required libraries

from flask import Flask, request, render_template,redirect,url_for,session
import numpy as np
import pandas as pd
from sklearn import metrics 
import warnings
warnings.filterwarnings('ignore')
from feature import generate_data_set
# Gradient Boosting Classifier Model
from sklearn.ensemble import GradientBoostingClassifier
import pickle
from matplotlib import pyplot
# split the dataset
from sklearn.model_selection import train_test_split

import mysql.connector
import os
# import pickle
# import numpy as np
import joblib
from flask_login import LoginManager,  login_user,logout_user





data = pd.read_csv("phishing.csv")
#droping index column
data = data.drop(['Index'],axis = 1)
# Splitting the dataset into dependant and independant fetature

X = data.drop(["class"],axis =1)
y = data["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=100)
# instantiate the model
gbc = GradientBoostingClassifier(max_depth=4,learning_rate=0.7)

# fit the model 
gbc.fit(X,y)
#pickle.dump(gbc,open('model.pkl','wb'))
print(gbc.score(X,y))
#predictions = gbc.predict(X_test)
# from sklearn.metrics import classification_report, confusion_matrix
# print(classification_report(y_test, predictions))
# print(confusion_matrix(y_test, predictions))
# c=confusion_matrix(y_test, predictions)



# import seaborn as sns
# import matplotlib.pyplot as plt
# ax = sns.heatmap(c, annot=True, cmap='Blues')

# ax.set_title('Confusion Matrix\n\n');
# ax.set_xlabel('\nPredicted Values')
# ax.set_ylabel('Actual Values ');

# ## Ticket labels - List must be in alphabetical order
# ax.xaxis.set_ticklabels(['False','True'])
# ax.yaxis.set_ticklabels(['False','True'])

# ## Display the visualization of the Confusion Matrix.
# plt.show()


# app = Flask(__name__)


app=Flask(__name__)
conn=mysql.connector.connect(host="127.0.0.1",user="root",password="",database="cybercrime")
cursor = conn.cursor()

@app.route('/')
def homes():
    return render_template('index.html')

@app.route('/home')
def index():
    return render_template('home.html')


@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/login_validation',methods=['GET','POST'])
def login_validation():
    username=request.form.get('username')
    password=request.form.get('password')


    cursor.execute("""SELECT * FROM `register` WHERE `username` LIKE '{}' AND `password` LIKE '{}'""".format(username,password,))
    users=cursor.fetchall()
    print(users)

    if len(users)>0:
        return render_template('home.html')
    else:
        return render_template('login.html')

    
    
@app.route('/add_user',methods=['GET','POST'])
def add_user():
    name=request.form.get ('name')
    email=request.form.get('email')
    phone=request.form.get('phone')
    username=request.form.get('username')
    password=request.form.get('password')
    confirm_password=request.form.get('confirm_password')
    
    
    cursor.execute("""INSERT INTO `register` (`name`,`email`,`phone`,`username`,`password`,`confirm_password`)VALUES('{}','{}','{}','{}','{}','{}')""".format ( name,email,phone, username, password,confirm_password))
    conn.commit()
    return render_template('login.html')

@app.route("/logout")
def logout():
    return render_template('index.html')

@app.route("/upload")
def upload():
    return render_template('upload.html')






# @app.route('/predict')
# def home():
#     return render_template('prediction.html')
# @app.route('/predict',methods=['post'])
# def predict():




@app.route("/predict")
def home():
    return render_template("prediction.html", xx= -1)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":

        url = request.form["url"]
        x = np.array(generate_data_set(url)).reshape(1,30) 
        y_pred =gbc.predict(x)[0]
        #1 is safe       
        #-1 is unsafe
        y_pro_phishing = gbc.predict_proba(x)[0,0]
        y_pro_non_phishing = gbc.predict_proba(x)[0,1]
        # if(y_pred ==1 ):
        pred = "It is {0:.2f} % safe to go ".format(y_pro_phishing*100)
        return render_template('prediction.html',xx =round(y_pro_non_phishing,2),url=url )
        # else:
        #     pred = "It is {0:.2f} % unsafe to go ".format(y_pro_non_phishing*100)
        #     return render_template('index.html',x =y_pro_non_phishing,url=url )
    return render_template("prediction.html", xx =-1)


if __name__ == "__main__":
    app.run(debug=True)