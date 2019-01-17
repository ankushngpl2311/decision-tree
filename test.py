import pandas as pd
import math
from tree import *

data= pd.read_csv('train.csv')
label ="left"
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
		infor = infor + (n/total)*entropy(i[0],i[1])
	return infor

def info_gain(entropy,info):
	gain = entropy - info
	return gain

def feature_select(data,features,value_dict):
	global label
	# cols=data.columns
	max_gain=0

	for f in features:
		positive_f= data.loc[data[label]==1]
		negative_f= data.loc[data[label]==0]
		lpos_f= len(positive_f)
		lneg_f= len(negative_f)
		entr_f= entropy(lpos_f,lneg_f)
		values= value_dict[f]
		positive_list =[]
		negative_list=[]
		for value in values:
			data2= data.loc[data[f]==value]
			positive=data2.loc[data2[label]==1]
			negative = data2.loc[data2[label]==0]
			lpos= len(positive)
			lneg = len(negative)
			# entr_v = entropy(lpos,lneg)
			positive_list.append(lpos)
			negative_list.append(lneg)

		information = info(positive_list,negative_list)

		gain = info_gain(entr_f,information)
		print("feature= ",f)
		print("gain= ",gain)
		if(gain>max_gain):
			f_selected = f
			max_gain=gain

	return f_selected




def make_tree(data,features,value_dict):
	f= feature_select(data,features,value_dict)
	temp = copy.deepcopy(features)
	temp.remove(f)
	







value_dict= prepare_dict(data)
print("value dict= ",value_dict)

# print("entropy s")
# print(entropy(9,5))
# print("entropy sunny")
# print(entropy(2,3))
# x=info([2,4,3],[3,0,2])
# print(x)
# print(info_gain(entropy(9,5),x))
features= data.columns
print(features)
features=features.drop(label)
# features=["outlook","temp","humidity","windy"]
x= feature_select(data,features,value_dict)
print("feature selected= ",x)
# print()