#importing required libraries

from flask import Flask, request, render_template
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

data = pd.read_csv("phishing.csv")
#droping index column
data = data.drop(['Index'],axis = 1)
# Splitting the dataset into dependant and independant fetature

X = data.drop(["class"],axis =1)
y = data["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=100)
# instantiate the model
gbc = GradientBoostingClassifier(max_depth=7,learning_rate=0.9)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()

# fit the model 
gbc.fit(X,y)
#pickle.dump(gbc,open('model.pkl','wb'))
print(gbc.score(X,y))
predictions = gbc.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
c=confusion_matrix(y_test, predictions)



import seaborn as sns
import matplotlib.pyplot as plt
ax = sns.heatmap(c, annot=True, cmap='Blues')

ax.set_title('Confusion Matrix\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()


app = Flask(__name__)


@app.route("/")
def index():
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