import pandas as pd
import math
# from tree import *
import copy
from sklearn.model_selection import train_test_split
import pickle as pk
from sklearn.metrics import precision_score



class node:
	def __init__(self):
		self.children ={}   # {low:pointer to low,high:pointer to high}
	def insert(self,name,value,positive,negative):
		self.name= name
		self.value=value
		self.positive=positive
		self.negative=negative
		

# print("jjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj")
data= pd.read_csv('train.csv')[0:1000]
label ="left"
true= 1
false=0
def prepare_dict(data):
	cols= data.columns
	d={}
	for i in cols:
		d[i]= set(data[i])
		# print("len= ")
		print(len(d[i]))

	for i in cols:
		s= d[i]
		l=list(s)
		l.sort()
		d[i]=l

	return d


def prob(fav,total):
    prob = fav/total
    return prob

def entropy(positive,negative):
    total = positive +negative
    prob_yes = prob(positive,total)
    prob_no = prob(negative,total)
    if(prob_yes != 0 and prob_no != 0):
    	entr = -prob_yes*math.log(prob_yes,2) - prob_no*math.log(prob_no,2)
    elif(prob_yes == 0):
    	entr = - prob_no*math.log(prob_no,2)
    else:
    	entr = -prob_yes*math.log(prob_yes,2)


    return entr


def info(positive,negative):
	total =0
	feature=[]

	for i in positive:
	    total = total +i
	    f= [i]
	    feature.append(f)

	for count,j in enumerate(negative):
	    total = total+j 
	    feature[count].append(j)
	# print(feature)
	# print(total)
	infor=0

	for i in feature:
		n= i[0] +i[1]  #positive +neg
		if(n==0):
			continue
		infor = infor + (n/total)*entropy(i[0],i[1])
	return infor

def info_gain(entropy,info):
	gain = entropy - info
	return gain

def feature_select(data,features,value_dict):
	global label
	# cols=data.columns
	max_gain=-1
	# print("features in fselect= ",features)

	for f in features:
		positive_f= data.loc[data[label]==true]
		negative_f= data.loc[data[label]==false]
		lpos_f= len(positive_f)
		lneg_f= len(negative_f)
		entr_f= entropy(lpos_f,lneg_f)
		values= value_dict[f]
		# positive_list =[]
		# negative_list=[]
		for value in values:
			positive_list=[]
			negative_list=[]
			# print("feature= ",f)
			# print("value= ",value)
			data2= data.loc[data[f] > value]
			# print("data2= ",data2)

			positive=data2.loc[data2[label]==true]
			negative = data2.loc[data2[label]==false]
			lpos= len(positive)
			lneg = len(negative)
			# entr_v = entropy(lpos,lneg)
			positive_list.append(lpos)
			negative_list.append(lneg)
			data3= data.loc[data[f] <= value]

			positive1=data3.loc[data3[label]==true]
			negative1 = data3.loc[data3[label]==false]
			lpos1= len(positive1)
			lneg1 = len(negative1)
			# entr_v = entropy(lpos,lneg)
			positive_list.append(lpos1)
			negative_list.append(lneg1)

		# print("positive list= ",positive_list)
		# print("negative list= ",negative_list)

			information = info(positive_list,negative_list)
			gain = info_gain(entr_f,information)
		# print("gain= ",gain)
		# print("feature= ",f)
			# print("gain= ",gain)
			if(gain<0):
				print("negative gainnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn")
			if(gain>max_gain):
				f_selected = f
				split=value
				max_gain=gain
				pos_selected= lpos_f
				neg_selected= lneg_f


	ret= [f_selected,split,pos_selected,neg_selected]

	return ret




def make_tree(data,features,value_dict,n):
	# if(len(features)==0):
	# 	print("len of feature ==0")
	# 	return
	# print("features= ",features)
	lenpos = len(data.loc[data[label]==true])
	lenneg = len(data.loc[data[label]==false])
	if(len(features)==0) or (lenpos == 0) or (lenpos ==1 and lenneg ==0) or (lenpos==0 and lenneg==1) or (lenneg == 0):
		print("len of pos= ",lenpos)
		print("len of neg= ",lenneg)

		if(lenpos==0 or lenneg >= lenpos):
			n.insert(0,"null",lenpos,lenneg)
		if(lenneg==0 or lenneg < lenpos):
			n.insert(1,"null,",lenpos,lenneg)
		# print("len of pos= ",lenpos)
		# print("len of neg= ",lenneg)
		return
	# print("data= ",data)
	# print("features after= ",features)
	# print("valuedic= ",value_dict)

	f= feature_select(data,features,value_dict)
	print("feature selected= ",f)
	# f= [f_selected,split,pos_selected,neg_selected]
	



	value = f[1]
	tempdict= copy.deepcopy(value_dict)
	# print(type(tempdict))
	# print(type(tempdict[f]))
	# print(tempdict)
	# print(tempdict[f])
	tempdict[f[0]].remove(f[1])
	
	temp = copy.deepcopy(features)
	if(len(tempdict[f[0]])==0):
		temp=temp.drop(f[0])

	rdict=copy.deepcopy(tempdict)

	for i in tempdict[f[0]]:
		if(i<value):
			rdict[f[0]].remove(i)


	ldict=copy.deepcopy(tempdict)

	for i in tempdict[f[0]]:
		if(i>=value):
			ldict[f[0]].remove(i)

	# print("ldict= ",ldict)
	# print("rdict= ",rdict)


	# n=node()
	n.insert(f[0],f[1],f[2],f[3])
	# print("f= ",f)
	# values = value_dict[f[0]]
	# print("data= ",data)
	# print("value dict= ",value_dict)
	# print("values= ",values)

	
	# print("value = ",value)
	# LEFT CHILD
	n1=node()
	n.children["low"]=n1
	data_tosend= data.loc[data[f[0]]<=value]
	# print("data2= \n",data_tosend)
	make_tree(data_tosend,temp,ldict,n1)
	
	#RIGHT CHILD
	n2=node()
	n.children["high"]=n2
	data_tosend2= data.loc[data[f[0]]>value]
	make_tree(data_tosend2,temp,rdict,n2)





def predict(x,n):
	y=[]
	print(type(n))
	# print("root name ",root.name)

	# for index,row in x.iterrows():
	row= x.iloc[0]
	print("row= ",row)

	for index,row in x.iterrows():
		root=n

		while(len(root.children) != 0):
			name = root.name
			print("name= ",name)

			val =row[name]
			print("val= ",val)
			compare= root.value
			if(val<= compare):
				root=root.children["low"]
			if(val> compare):
				root= root.children["high"]
			print("root.name= ",root.name)
			print("row value= ",)
			# print("root name ",root.name)
			# print("root children ",root.children)

			# root = root.children[val]
		# print("outside loop= ",root.name)
		# print("rootout= ",root)
		print("root.name= ",root.name)
		y.append(root.name)
		print("y[0]=",y[0])

	# print(y)
	return y

		

def accuracy(ytest,ypredict):
	c=0
	l =len(ytest)

	for count,i in enumerate(ytest):
		if(ytest[count] == ypredict[count]):
			c= c +1
	print("count= ",c)
	print("total= ",l)

	return c/l



def parameters(ytest,ypredict):

	tp=0
	fp=0
	fn=0
	for count,i in enumerate(ytest):
		if(ypredict[count]==true and ytest[count]==true):
			tp= tp+1
		if(ypredict[count]==true and ytest[count]==false):
			fp = fp+1
		if(ypredict[count]==false and ytest[count]== true):
			fn = fn +1
	l=[tp,fp,fn]
	print("tp fp fn= ",l)

	return l


def recall(tp,fn):
	den = tp+fn
	if(den==0):
		return 0
	return tp/den

def precision(tp,fp):
	den= tp+fp
	if(den==0):
		return 0
	return tp/den

def f1_score(precision,recall):
	if(precision != 0):
		pr1= 1/precision
	else:
		pr1=0
	if(recall!=0):
		rc1= 1/recall
	else:
		rc1=0

	tot= pr1+rc1
	if(tot!=0):
		ret = 2/tot
	else:
		ret=0

	return ret




data2=data
print("data columns= ",data.columns)
# data2=[list(x) for x in data.values]
columns= ['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident', 'left',
       'promotion_last_5years', 'sales', 'salary']

categorical=['sales', 'salary']
cat_list=[]
for j in categorical:
	d={}
	for count,i in enumerate(set(data[j])):
		d[i]=count
	cat_list.append(d)
print(cat_list)








# print("data2= ")
# print(data2)
# print("typedata2 = ",type(data2))

# print(type(data2[0]))
# print("data2[0]= ",data2[0])

# [{'support': 0, 'product_mng': 1, 'accounting': 2, 'IT': 3, 'hr': 4, 'marketing': 5, 'RandD': 6, 
# 'management': 7, 'sales': 8, 'technical': 9}, {'medium': 0, 'high': 1, 'low': 2}]

# print(data2.columns())

# for col in data2.columns():
# data2.loc[data2["sales"] == 'support', col] = 0
# data2.loc[data2["sales"] == 'product_mng', col] = 1
# data2.loc[data2["sales"] == 'accounting', col] = 2
# data2.loc[data2["sales"] == 'IT', col] = 3
# data2.loc[data2["sales"] == 'hr', col] = 4
# data2.loc[data2["sales"] == 'marketing', col] = 5
# data2.loc[data2["sales"] == 'RandD', col] = 6
# data2.loc[data2["sales"] == 'management', col] = 7
# data2.loc[data2["sales"] == 'sales', col] = 8
# data2.loc[data2["sales"] == 'technical', col] = 9
# data2.loc[data2["sales"] == 'medium', col] = 0
# data2.loc[data2["sales"] == 'high', col] = 1
# data2.loc[data2["sales"] == 'low', col] = 2


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
# if(data2.loc[data2["sales"] == 'support']):
# 	data2["sales"]=0
# if(data2.loc[data2["sales"] == 'product_mng']):
# 	data2["sales"]=1
# data2.loc[data2["sales"] == 'accounting'] = 2
# data2.loc[data2["sales"] == 'IT'] = 3
# data2.loc[data2["sales"] == 'hr'] = 4
# data2.loc[data2["sales"] == 'marketing'] = 5
# data2.loc[data2["sales"] == 'RandD'] = 6
# data2.loc[data2["sales"] == 'management'] = 7
# data2.loc[data2["sales"] == 'sales'] = 8
# data2.loc[data2["sales"] == 'technical'] = 9
# data2.loc[data2["salary"] == 'medium'] = 0
# data2.loc[data2["salary"] == 'high'] = 1
# data2.loc[data2["salary"] == 'low'] = 2

# for row in data2.iterrows():
# 	print("len row= ",len(row))
# 	print("row= ",row)
	# if(row[8]=='support'):
	# 	row[8]=0
	# elif(row["salary"]=='product_mng'):
	# 	row["salary"]=1
	# elif(row["salary"]=='accounting'):
	# 	row["salary"]=2
	# elif(row["salary"]=='IT'):
	# 	row["salary"]=3
	# elif(row["salary"]=='hr'):
	# 	row["salary"]=4
	# elif(row["salary"]=='marketing'):
	# 	row["salary"]=5
	# elif(row["salary"]=='RandD'):
	# 	row["salary"]=6
	# elif(row["salary"]=='management'):
	# 	row["salary"]=7
	# elif(row["salary"]=='sales'):
	# 	row["salary"]=8
	# elif(row["salary"]=='technical'):
	# 	row["salary"]=9
	# if(row["salary"]=='medium'):
	# 	row["salary"]=0
	# elif(row["salary"]=='high'):
	# 	row["salary"]=1
	# elif(row["salary"]=='low'):
	# 	row["salary"]=2





print("data2= ",data2)
for i in cat_list:
	data2=data2.replace(i)
print(data2)
value_dict= prepare_dict(data2)
print("value dict= ",value_dict)
features = data2.columns
features=features.drop(label)
print("features= ",features)
# print("data2= ",data2)
x=data2.drop(columns=label)
y= data2[label]
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2)
l= [xTrain,yTrain]
data_train=pd.concat(l,axis=1)
# print("data train= ",data_train)
l=[xTest,yTest]
data_test = pd.concat(l,axis=1)




# print("entropy s")
# print(entropy(9,5))
# print("entropy sunny")
# print(entropy(2,3))
# x=info([2,4,3],[3,0,2])
# print(x)
# print(info_gain(entropy(9,5),x))
# features= data.columns
# print(features)
# features=features.drop(label)


# features=["outlook","temp","humidity","windy"]
# x= feature_select(data,features,value_dict)
# print("feature selected= ",x)
# print()



root= node()
make_tree(data_train,features,value_dict,root)






########################
#DUMP MODEL
#####################
# f= open("modelentropy.pkl","wb")
# pk.dump(root,f)
# f.close()
###########################
###########################
# print("postorder")
# print(postorder(root))
########################
#LOAD MODEL
#####################
# f= open("modelentropy.pkl","rb")
# root=pk.load(f)
# f.close()
###########################
###########################
print("type root= ",type(root))
ypredict=predict(xTest,root)
yTest= yTest.tolist()
# print(yTest)
# print("ypredict= ",ypredict)
# print("type= ",type(yTest[0]))
# print("ytest[0] ",yTest[0])
# print("ypredict [0] ",ypredict[0])
print("accuracy= ",accuracy(yTest,ypredict))

l= parameters(yTest,ypredict)
pr=precision(l[0],l[1])
rc=recall(l[0],l[2])
print("recall= ",rc)
print("precision= ",pr)
print("f1 score= ",f1_score(pr,rc))

print("sklearn precision= ",precision_score(yTest,ypredict))

