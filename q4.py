import pandas as pd
import math
# from tree import *
import copy
from sklearn.model_selection import train_test_split
import pickle as pk
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt


class node:
	def __init__(self):
		self.children ={}   # {low:pointer to low,high:pointer to high}
	def insert(self,name,value,positive,negative):
		self.name= name
		self.value=value
		self.positive=positive
		self.negative=negative
		


data= pd.read_csv('train.csv')
label ="left"
true= 1
false=0



data2=data
print("data columns= ",data.columns)
# data2=[list(x) for x in data.values]
columns= ['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident', 'left',
       'promotion_last_5years', 'sales', 'salary']



salary= data2["salary"]
for i in salary:
	if(i=="medium"):
		i=0
	elif(i=="high"):
		i=1
	elif(i=="low"):
		i=2
sales= data2["sales"]

for i in sales:
	if(i=='support'):
		i=0
	elif(i=='product_mng'):
		i=1
	elif(i=='accounting'):
		i=2
	elif(i=='IT'):
		i=3
	elif(i=='hr'):
		i=4
	elif(i=='marketing'):
		i=5
	elif(i=='RandD'):
		i=6
	elif(i=='management'):
		i=7
	elif(i=='sales'):
		i=8
	elif(i=='technical'):
		i=9

# print("data2= ",data2)
x=data2
y= data2[label]
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2)
l= [xTrain,yTrain]
data_train=pd.concat(l,axis=1)
# print("data train= ",data_train)
l=[xTest,yTest]
data_test = pd.concat(l,axis=1)



xyes=xTrain.loc[xTrain["left"]==1]
xyes= xyes["satisfaction_level"]

xno=xTrain.loc[xTrain["left"]==0]
xno= xno["satisfaction_level"]

yyes=xTrain.loc[xTrain["left"]==1]
yyes= yyes["number_project"]

yno=xTrain.loc[xTrain["left"]==0]

yno =yno["number_project"]

plt.scatter(xyes, yyes, c='red')
plt.scatter(xno,yno,c='blue')
plt.title('Scatter plot x_yes y_yes')
plt.xlabel('satisfaction level')
plt.ylabel('number project')
plt.show()
