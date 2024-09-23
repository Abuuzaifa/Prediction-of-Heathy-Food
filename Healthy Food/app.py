from flask import Flask,render_template,url_for,request
from flask_material import Material
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import plotly
import plotly.graph_objs as go
import json
from sklearn.naive_bayes import GaussianNB
from flask import flash



# EDA PKg
import pandas as pd 
import numpy as np 

# ML Pkg


app = Flask(__name__)
Material(app)
app.secret_key="dont tell any one"
@app.route('/')
def index():
    return render_template("index.html")


@app.route('/',methods=["POST"])
def login():
    if request.method == 'POST':
        username = request.form['id']
        password = request.form['pass']
        if username=='admin' and password=='admin':
            return render_template("main.html")
        else:
            flash("wrong password")
            return render_template("index.html")
@app.route('/main',methods=["POST"])
def analyze():
	if request.method == 'POST':
		age = request.form['age']
		gender = request.form['gender']
		weight = request.form['weight']
		blood_group = request.form['blood_group']
		blood_pressure = request.form['bp']
		suger = request.form['sugar']
		red_blood_cell = request.form['red_blood_cell']
		White_blood_Cells = request.form['white_blood_cell']
		platelets=request.form['platelets']
		haemo = request.form['haemo']
		protein = request.form['protein']

		age2=0
		weight2=0
		blood_pressure2=0
		suger2=0
		haemo2=0
		platelets2=0
		red_blood_cell2=0
	

		# Clean the data by convert from unicode to float 
		sample_data = [gender,age,weight,blood_group,blood_pressure,haemo,suger,platelets,White_blood_Cells,red_blood_cell]
		clean_data = [float(i) for i in sample_data]
		age1=int(age)
		weight1=int(weight)
		blood_pressure1=int(blood_pressure)
		suger1=int(suger)
		haemo1=float(haemo)

		red_blood_cell1=float(red_blood_cell)

		if age1<18 or age1>50:
			age2=age1
		elif weight1<50:
			weight2=weight1
		elif blood_pressure1>80:
			blood_pressure2=blood_pressure1
		elif suger1>110:
			suger2=suger1
		elif haemo1<13.5 and gender==['1']:
			haemo2=haemo1
		elif haemo1<12 and gender==['0']:
			haemo2=haemo1

		elif red_blood_cell1<3.5:
			red_blood_cell2=red_blood_cell1

		


	

		# Reshape the Data as a Sample not Individual Features
		ex1 = np.array(clean_data).reshape(1,-1)
	
		print(ex1)

		# ex1 = np.array([6.2,3.4,5.4,2.3]).reshape(1,-1)
		data = pd.read_csv('healthfood.csv')
		data=data.drop(columns=['name','id'])
		data=data.replace(['female','male','Healthy','Not Healthy','A+','A-','B+','B-','AB+','AB-','O+','O-'],['0','1','0','1','0','1','2','3','4','5','6','7'])

		X = data.drop(columns=['eligibility'])
		y = data['eligibility'].values

		X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size = 0.2, random_state=1, stratify=y)
		
		
		gnb = GaussianNB()
		gnb.fit(X_train1, y_train1)


		knn = KNeighborsClassifier(n_neighbors = 3)
		knn.fit(X_train1,y_train1)

		clf =DecisionTreeClassifier(criterion="entropy",random_state=2,max_depth=5)
		clf.fit(X_train1, y_train1)


		
                
		result=gnb.predict(ex1)
		if result=='1':
			class1="Not Healthy person"
		else:
			class1="Healthy person "	
		Y_pred_nb =gnb.predict(X_test1)
		Y_pred_nb.shape
		score_nb = round(accuracy_score(Y_pred_nb,y_test1)*100,2)
           
		

           
		
		result=knn.predict(ex1)
		if result=='1':
			class3="Not Healthy person"
		else:
			class3="Healthy person "	
		Y_pred_knn =knn.predict(X_test1)
		Y_pred_knn.shape
		score_knn = round(accuracy_score(Y_pred_knn,y_test1)*100,2)
            
		
		result=clf.predict(ex1)
		if result=='1':
			class4="Not Healthy person"
		else:
			class4="Healthy person "	
		Y_pred_dt =clf.predict(X_test1)
		Y_pred_dt.shape
		score_dt = round(accuracy_score(Y_pred_dt,y_test1)*100,2)




		df1 = pd.read_excel('bloodupdated.xlsx')

		X = df1.drop(columns=['class'])
		y = df1['class'].values
		X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size = 0.2, random_state=1, stratify=y)
		gfr=100

		sample_data1 = [age,gender,weight,blood_group,blood_pressure,suger,red_blood_cell,White_blood_Cells,haemo,protein]
		clean_data1 = [float(i) for i in sample_data1]
		ex2 = np.array(clean_data1).reshape(1,-1)


		def predictedFood(res):
			if res==1:
				p_class=["apple","egg","yogurt","sunflower","seeds"]
			elif res==2:
				p_class=["EGGS", "Beans", "Greens", "Brocolli"]
			elif res==3:
				p_class=["Bitroot","Pomegranate","Dates","Banana"]
			elif res==4:
				p_class=["Eggs", "Kidney beans","Brown","rice Banana"]
			elif res==5:
				p_class=["Citrus", "fruits", "Tomatoes", "Dark green leafy", "vegetables", "Berries"]
			elif res==6:
				p_class=["Seafood", "Fruit" ,"juice", "Milk", "Brown Rice"]
			elif res==7:
				p_class=["Eggs","Kidney beans", "Wheat", "Bitroot"]
			elif res==8:
				p_class=["EGGS", "Beans","Tomatoes", "Dark green leafy"]
			elif res==9:
				p_class=["Miomatoes", "Dark green leafy", "vegetables", "Berriesllets"]
			else:
				p_class=["Pomegranate", "EGGS", "Beans"]
			return p_class
		
		
		
		clf1 = tree.DecisionTreeClassifier()
		clf1.fit(X_train1, y_train1)


		result=clf1.predict(ex2)
		class6=predictedFood(result)            

	return render_template('view.html', gender=gender,
		age=age,
		weight=weight,
		blood_group=blood_group,
		haemo=haemo,
		suger=suger,
        white_blood_cell=White_blood_Cells,
        red_blood_cell=red_blood_cell,
        result=result,
		age2=age2,
		weight2=weight2,
		blood_pressure2=blood_pressure2,
		suger2=suger2,
		haemo2=haemo2,
		platelets2=platelets2,
		red_blood_cell2=red_blood_cell2,class1=class1,score_nb=score_nb,class3=class3,score_knn=score_knn,class4=class4,score_dt=score_dt,class6=class6)



if __name__ == '__main__':
	app.run(debug=True)
