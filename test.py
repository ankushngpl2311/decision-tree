import pandas as pd
import math
from tree import *
import copy
from sklearn.model_selection import train_test_split

data= pd.read_csv('train.csv')
label ="left"
true= 1
false=0
def prepare_dict(data):
	cols= data.columns
	d={}
	for i in cols:
		d[i]= set(data[i])

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
	print("features in fselect= ",features)

	for f in features:
		positive_f= data.loc[data[label]==true]
		negative_f= data.loc[data[label]==false]
		lpos_f= len(positive_f)
		lneg_f= len(negative_f)
		entr_f= entropy(lpos_f,lneg_f)
		values= value_dict[f]
		positive_list =[]
		negative_list=[]
		for value in values:
			data2= data.loc[data[f]==value]
			# print("data2= ",data2)

			positive=data2.loc[data2[label]==true]
			negative = data2.loc[data2[label]==false]
			lpos= len(positive)
			lneg = len(negative)
			# entr_v = entropy(lpos,lneg)
			positive_list.append(lpos)
			negative_list.append(lneg)
		# print("positive list= ",positive_list)
		# print("negative list= ",negative_list)

		information = info(positive_list,negative_list)

		gain = info_gain(entr_f,information)
		print("gain= ",gain)
		# print("feature= ",f)
		# print("gain= ",gain)
		if(gain>max_gain):
			f_selected = f
			max_gain=gain
			pos_selected= lpos_f
			neg_selected= lneg_f


	ret= [f_selected,pos_selected,neg_selected]

	return ret




def make_tree(data,features,value_dict,n):
	# if(len(features)==0):
	# 	print("len of feature ==0")
	# 	return
	# print("features= ",features)
	lenpos = len(data.loc[data[label]==true])
	lenneg = len(data.loc[data[label]==false])
	if(len(features)==0) or (lenpos == 0) or (lenpos ==1 and lenneg ==0) or (lenpos==0 and lenneg==1) or (lenneg == 0):
		if(lenpos==0 or lenneg > lenpos):
			n.insert(0,lenpos,lenneg)
		if(lenneg==0 or lenneg <= lenpos):
			n.insert(1,lenpos,lenneg)
		print("len of pos= ",lenpos)
		print("len of neg= ",lenneg)
		return
	# print("data= ",data)
	print("features after= ",features)
	# print("valuedic= ",value_dict)

	f= feature_select(data,features,value_dict)
	print("feature selected= ",f)
	temp = copy.deepcopy(features)
	temp=temp.drop(f[0])
	# n=node()
	n.insert(f[0],f[1],f[2])
	# print("f= ",f)
	values = value_dict[f[0]]
	# print("data= ",data)
	# print("value dict= ",value_dict)
	print("values= ",values)

	for value in values:
		print("value = ",value)
		n1=node()
		n.children[value]=n1
		data_tosend= data.loc[data[f[0]]==value]
		# print("data2= \n",data_tosend)
		make_tree(data_tosend,temp,value_dict,n1)



def predict(x,root):
	y=[]
	print(type(root))
	# print("root name ",root.name)

	for index,row in x.iterrows():
		
		# print("row= ",row)

		while(len(root.children) != 0):
			name = root.name
			val =row[name]
			print("root name ",root.name)
			print("root children ",root.children)
			root = root.children[val]
		# print("outside loop= ",root.name)
		print("rootout= ",root)
		y.append(root.name)

	# print(y)
	return y

		

def accuracy(ytest,ypredict):
	count=0
	l =len(ytest)

	for count,i in enumerate(ytest):
		if(ytest[count] == ypredict[count]):
			count= count +1
	print("count= ",count)
	print("total= ",l)

	return count/l










data2= data[['Work_accident','promotion_last_5years', 'sales', 'salary','left']]
value_dict= prepare_dict(data2)
print("value dict= ",value_dict)
features = data2.columns
features=features.drop(label)
print("features= ",features)
# print("data2= ",data2)
x=data2.drop(columns=label)
y= data2[label]
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)
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
# print("postorder")
# print(postorder(root))
ypredict=predict(xTest,root)
yTest= yTest.tolist()
print(yTest)
print("type= ",type(yTest))
print("ytest[0] ",yTest[0])
print("ypredict [0] ",ypredict[0])
print("accuracy= ",accuracy(yTest,ypredict))