from __future__ import division
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import random
from pprint import pprint
from sklearn.model_selection import train_test_split



data = pd.read_csv("train.csv")
data=data.rename(columns={"left":"label"})





#function to find out whether impurity is there or not 
def purity(data,pure=1):
    pure=1
    label_col=data[:,-1]
    uni_val=np.unique(label_col)
    pure=1
    if len(uni_val) == 1:
        return True
    else:
        return False
    
#classification based on categories in the label that occurs the most number of times    


def all_splits(data):
    
    potential_splits = {}
    _, n_columns = data.shape
    for column_index in range(n_columns - 1):          # excluding the last column which is the label
        values = data[:, column_index]
        unique_values = np.unique(values)
        
        type_of_feature = FEATURE_TYPES[column_index]
        if type_of_feature == "continuous":
            potential_splits[column_index] = []
            for index in range(len(unique_values)):
                if index != 0:
                    current_value = unique_values[index]
                    previous_value = unique_values[index - 1]
                    potential_split = (current_value + previous_value) / 2

                    potential_splits[column_index].append(potential_split)
        
        # feature is categorical
        # (there need to be at least 2 unique values, otherwise in the
        # split_data function data_below would contain all data points
        # and data_above would be empty)
        elif len(unique_values) > 1:
            potential_splits[column_index] = unique_values
    
    return potential_splits

def classify(data):
    label_col=data[:,-1]
    uni_val,counts_uni_val=np.unique(label_col,return_counts=True)
    index=counts_uni_val.argmax()
    classification=uni_val[index]
    return classification

def split_data(data, split_column, split_value):
    split_column_values = data[:, split_column]

    type_of_feature = FEATURE_TYPES[split_column]
    if type_of_feature == "continuous":
        belowdata = data[split_column_values <= split_value]
        abovedata = data[split_column_values >  split_value]
    
    # feature is categorical   
    else:
        belowdata = data[split_column_values == split_value]
        abovedata = data[split_column_values != split_value]
    
    return belowdata, abovedata

#function for finding the entropy
def calculate_entropy(data,data1='None'):
    lables = data[:, -1]
    counts = {}
    for label in lables:
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    entropy = 1
    true_probab = counts[label] / float(len(lables))
    not_true_probab = 1-true_probab
    if true_probab == 0.0 or not_true_probab == 0.0:
        return 0.0
    first=((-true_probab) * math.log(true_probab, 2))
    second=( not_true_probab * math.log(not_true_probab, 2))
    entropy=first-second
    #entropy = ((-true_probab) * math.log(true_probab, 2)) - ( not_true_probab * math.log(not_true_probab, 2))
    return entropy 

#function for finding the overall entropy or information gain
def calculate_overall_entropy(belowdata, abovedata, curr_entropy,data1='None'):
    p = float(len(belowdata)) / (len(belowdata) + len(abovedata))
   # new_entropy = (p*calculate_entropy(belowdata)) + ((1-p)*calculate_entropy(abovedata))
    x=(p*calculate_entropy(belowdata))
    y=((1-p)*calculate_entropy(abovedata))
    new_entropy=x+y
    information_gain = curr_entropy - new_entropy
    return information_gain

def infogain(belowdata, abovedata, curr_entropy,data1='None'):
    p = float(len(belowdata)) / (len(belowdata) + len(abovedata))
    x=(p*calculate_entropy(belowdata))
    y=((1-p)*calculate_entropy(abovedata))
    new_entropy=x+y
    information_gain = curr_entropy - new_entropy
    return information_gain



# =============================================================================
# def calculate_entropy(data):
#     
#     label_column = data[:, -1]
#     _, counts = np.unique(label_column, return_counts=True)
# 
#     probabilities = counts / counts.sum()
#     entropy = sum(probabilities * -np.log2(probabilities))
#      
#     return entropy
# 
# def calculate_overall_entropy(data_below, data_above):
#     
#     n = len(data_below) + len(data_above)
#     p_data_below = len(data_below) / n
#     p_data_above = len(data_above) / n
# 
#     overall_entropy =  (p_data_below * calculate_entropy(data_below) 
#                       + p_data_above * calculate_entropy(data_above))
#     return overall_entropy
# =============================================================================

#function for finding the best splits out of all possible splits
def determine_best_split(data, potential_splits):
    best_information_gain = 0
    best_split_column = best_split_value = None
    curr_impurity = calculate_entropy(data)
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            # partition
            belowdata, abovedata = split_data(data, split_column=column_index, split_value=value)
            
            # information gain
            current_overall_entropy = infogain(belowdata, abovedata, curr_impurity)
            #print(current_overall_entropy)
            if current_overall_entropy >= best_information_gain:
                best_information_gain = current_overall_entropy
                best_split_column = column_index
                best_split_value = value
    
    return best_split_column, best_split_value

def determine_type_of_feature(df):
    n_unique_values_treshold = 15
    feature_types = []
    
    for feature in df.columns:
        if feature != "label":
            unique_values = df[feature].unique()
            example_value = unique_values[0]

            if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_treshold):
                feature_types.append("categorical")
            else:
                feature_types.append("continuous")
    
    return feature_types

default_depth=10
default_samples=2
def make_tree(df, counter=0, min_samples=default_samples, max_depth=default_depth):
    
    # data preparations
    if counter == 0:
        global COLUMN_HEADERS, FEATURE_TYPES
        COLUMN_HEADERS = df.columns
        FEATURE_TYPES = determine_type_of_feature(df)
        data = df.values
    else:
        data = df           
    
    
    # base cases
    if (purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        classification = classify(data)
        
        return classification

    
    # recursive part
    else:    
        counter += 1

       
        potential_splits = all_splits(data)
        split_column, split_value = determine_best_split(data, potential_splits)
        belowdata, abovedata = split_data(data, split_column, split_value)
        
        feature_name = COLUMN_HEADERS[split_column]
        type_of_feature = FEATURE_TYPES[split_column]
        #for continuos features
        if type_of_feature == "continuous":
            question = "{} <= {}".format(feature_name, split_value)
            
        #for categorical features
        else:
            question = "{} = {}".format(feature_name, split_value)
        
        # instantiate sub-tree
        sub_tree = {question: []}
        
        # find answers (recursion)
        yes_answer = make_tree(belowdata, counter, min_samples, max_depth)
        no_answer = make_tree(abovedata, counter, min_samples, max_depth)
        
        # If the answers are the same, then there is no point in asking the qestion.
        # This could happen when the data is classified even though it is not pure
        # yet (min_samples or max_depth base case).
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)
        
        return sub_tree



def classify_example(example, tree):
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split(" ")

    # ask question
    if comparison_operator == "<=":
        if example[feature_name] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]
    
    # feature is categorical
    else:
        if str(example[feature_name]) == value:
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    # base case
    if not isinstance(answer, dict):
        return answer
    
    # recursive part
    else:
        residual_tree = answer
        return classify_example(example, residual_tree)

#classify_example(example, tree)
     
        
def final_parameters(df, tree):

    df["classification"] = df.apply(classify_example, args=(tree,), axis=1)
    df["classification_correct"] = df["classification"] == df["label"]
    tp=0  
    fp=0 
    fn=0
    predicted = list()
    predicted = df["classification"].values
    actual = list()
    actual = df["label"].values
    for i in range(0,len(predicted)):
        if actual[i] == predicted[i] and actual[i] ==1:
            tp += 1
        elif actual[i] != predicted[i] and predicted[i] == 1:
            fp += 1
        elif actual[i] ==1 and predicted[i] ==0:
            fn += 1
            
    deno1=tp+fp
    deno2=tp+fn
    precision=float(tp/deno1)
    recall=float(tp/deno2)
    print("Precision is :",precision)
    print("Recall is :",recall)
    deno3=precision+recall
    num3=2*precision*recall
    f1score=num3/deno3
    print("F1 score is :",f1score)
          
    
    accuracy = df["classification_correct"].mean()
    
    return accuracy

data_train,data_test=train_test_split(data,test_size=0.2)
data = data_train.values
root = make_tree(data_train, max_depth=10)
pprint(root)

example = data_test.iloc[0]

accuracy = final_parameters(data_test, root)
print('Accuracy is:')
print(accuracy)

