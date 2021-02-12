from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.externals import joblib


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	import numpy as np # linear algebra
	import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
	from sklearn.metrics import confusion_matrix,accuracy_score
	from sklearn.model_selection import train_test_split
	from sklearn.preprocessing import StandardScaler
	from sklearn.metrics import classification_report

	#Loading the data
	heart_data= pd.read_csv("heartdata.csv", encoding="latin-1")
	
	#Removing unknown or NULL or Missing values
	heart_data.rename(columns={'num       ': 'target'}, inplace=True) 
	heart_data=heart_data.replace('?',None)
	heart_data=heart_data.replace('?',0)

	
	
	# Features and Labels
	from sklearn.model_selection import train_test_split
	#heart_data['label'] = heart_data['class'].map({'lowRisk': 0, 'highRisk': 1})
	y = heart_data["target"]
	X = heart_data.drop('target',axis=1)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 0)
	
	#Normalizing the data
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)

	# Model learning
	#Logistic Regression 
	from sklearn.linear_model import LogisticRegression

	logistic = LogisticRegression()
	model=logistic.fit(X_train,y_train)
	logistic.score(X_test,y_test)
	

	if request.method == 'POST':
		heartdata = request.form['heartdata']
		data = [heartdata]
		vect = scaler.fit_transform(data).toarray()
		my_prediction = model.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)