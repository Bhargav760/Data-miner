import streamlit as st
import pandas as pd 
import math
import seaborn as sns
from collections import Counter
from sklearn import tree, preprocessing
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import scipy.stats as stats
from sklearn.metrics import confusion_matrix 
from numpy import *
import numpy as np
from joblib.numpy_pickle_utils import xrange
import itertools
from scipy.stats import chi2_contingency
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text, DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

st.title("Mini project DM lab")

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file :
    df  = pd.read_csv(uploaded_file)


    # frequency = {}
    # for i in df['EmployeeName']:
    #     frequency.setdefault(i, 0)
    #     frequency[i]+=1 
    # st.write(frequency)
    t = st.write(df.iloc[:2000,:])
    related_att = []
    columns = df.columns
    alpha = 0.05
    # for column in columns[1:-1]:
    #     data = [df[column],df['is_promoted']]
        # stat, p, dof, expected = chi2_contingency(data)
        # if p<=alpha:
        #     related_att.append(column)
        #     correlation_coefficient = df["is_promoted"].corr(df[column])
        #     st.write(column,"is responsible for promotion with correlation coefficient = ",correlation_coefficient)
        #     if(correlation_coefficient>0):
        #         st.write("This column is postively correlated")
        #     else :
        #         st.write("This column is negatively correlated")
        #     plot = plt.scatter(column,df['is_promoted'])
        #     st.pyplot(plt)

        # st.write("Chi square test")

        # xlabel = "is_promoted"
        # for column in df.columns[1:-1]:

        #     ylabel = column
        #     plt.locator_params(nbins = 10)

            
        #     contigency_table = pd.crosstab(df[xlabel],df[ylabel],margins=True,margins_name="All")
        #     st.text(contigency_table)
        #     rows = df[xlabel].unique()
        #     columns1 = df[ylabel].unique()
        #     print(columns1)
        #     chi_square = 0.0
        #     for i in columns1:
        #         for j in rows:
        #             obs = contigency_table[i][j]
        #             expected = (contigency_table[i]['All'] * contigency_table['All'][j])/(contigency_table['All']['All'])
        #             chi_square = chi_square + ((obs - expected)**2/expected)
        #     p_value = 1 - stats.chi2.cdf(chi_square,(len(rows) - 1)*(len(columns1) - 1))
        #     dof = (len(columns1) - 1)*(len(rows) - 1)

            
        #     if(p_value<chi_square):
        #         st.subheader("chi-square value")
        #         st.write(chi_square)
        #         st.subheader("degree of freedom")
        #         st.write(dof)
        #         st.subheader("columns have the corealation")
        #     else:
        #         st.subheader("columns don't have any relation")
        

    #scatter plots

    att1 = st.selectbox("First attribute",df.columns)
    att2 = st.selectbox("Second attribute",df.columns)

    
    plt.scatter(df[att1],df[att2], c ="black",s=2)
    plt.xlabel(att1)
    plt.ylabel(att2)
    st.write("Scatter Plot")
    st.pyplot(plt)
    
    plt.clf()
    
    #box plots
    fig = px.box(df, x=att1, y=att2, points="all")
    st.plotly_chart(fig)
    # fig.show()


    #density plot
    
    att3 = st.selectbox("Select attribute",df.columns)
    fig = plt.figure(figsize=(10, 4))
    sn = sns.kdeplot(df[att3], color='red', fill=True, alpha=.3, linewidth=0)
    st.pyplot(fig)

    # def splitdataset(balance_data):
  
    # # Separating the target variable
    #     X = balance_data.values[:, 1:-1]
    #     Y = balance_data.values[:, -1]
    
    #     # Splitting the dataset into train and test
    #     X_train, X_test, y_train, y_test = train_test_split( 
    #     X, Y, test_size = 0.3, random_state = 100)
        
    #     return X, Y, X_train, X_test, y_train, y_test
      
    # # Function to perform training with giniIndex.
    # def train_using_gini(X_train, X_test, y_train):
    
    #     # Creating the classifier object
    #     clf_gini = DecisionTreeClassifier(criterion = "gini",
    #             random_state = 100,max_depth=3, min_samples_leaf=5)
    
    #     # Performing training
    #     clf_gini.fit(X_train, y_train)
    #     return clf_gini
        
    # # Function to perform training with entropy.
    # def tarin_using_entropy(X_train, X_test, y_train):
    
    #     # Decision tree with entropy
    #     clf_entropy = DecisionTreeClassifier(
    #             criterion = "entropy", random_state = 100,
    #             max_depth = 3, min_samples_leaf = 5)



    #     clf_entropy.fit(X_train, y_train)
    #     return clf_entropy
    
    
    # # Function to make predictions
    # def prediction(X_test, clf_object):
    #     st.write("Results Using Entropy:")
    #     # Predicton on test with giniIndex
    #     y_pred = clf_object.predict(X_test)
    #     # print("Predicted values:")
    #     # print(y_pred)
    #     return y_pred
        
    # # Function to calculate accuracy
    # def cal_accuracy(y_test, y_pred):
        
    #     st.subheader("Confusion Matrix")
    #     st.write(confusion_matrix(y_test, y_pred))
    #     st.subheader("Accuracy")
    #     st.write(accuracy_score(y_test,y_pred)*100)
        
    #     st.subheader("Report ")
    #     st.write(classification_report(y_test, y_pred))

    # for i in columns[1:]:
    #     temp=np.unique(df[i])
    #     mp = {}
    #     count = 0
    #     for j in temp:
    #         mp[j] = count
    #         count+=1

    #     lst  = list(mp[val] for val in df[i])
    #     df[i] = lst
        

    # # st.write(df)
    # X, Y, X_train, X_test, y_train, y_test = splitdataset(df)
    # #clf_gini = train_using_gini(X_train, X_test, y_train)
    # clf_entropy = tarin_using_entropy(X_train, X_test, y_train)
    
    # # Operational Phase
    # # print("Results Using Gini Index:")
    
    # # Prediction using gini
    # #y_pred_gini = prediction(X_test, clf_gini)
    # #cal_accuracy(y_test, y_pred_gini)
    
    
    # # Prediction using entropy
    # y_pred_entropy = prediction(X_test, clf_entropy)
    

    # y_test_train = pd.DataFrame(columns=["y_actual","y_predicted"])
    # y_test_train['y_actual'] = y_test
    # y_test_train['y_predicted'] = y_pred_entropy
    # st.write(y_test_train)
    # cal_accuracy(y_test, y_pred_entropy)

    # tree.plot_tree(decision_tree,ax=ax,feature_names=features)
    # # tree_rules = export_text(model, feature_names=list(X_train.columns))
    # # print(columns[1:-1].tolist())
    

    targetAttr='is_promoted'       
    st.header("Decision Tree")
    data=df
    features = list(columns[1:])
    features.remove(targetAttr)

    def entropy(labels):
        entropy=0
        label_counts = Counter(labels)
        for label in label_counts:
            prob_of_label = label_counts[label] / len(labels)
            entropy -= prob_of_label * math.log2(prob_of_label)
        return entropy

    def information_gain(starting_labels, split_labels):
        info_gain = entropy(starting_labels)
        ans=0
        for branched_subset in split_labels:
            ans+=len(branched_subset) * entropy(branched_subset) / len(starting_labels)
        st.write("entropy:",ans)
        info_gain-=ans
        return info_gain

    def split(dataset, column):
        split_data = []
        col_vals = data[column].unique()
        for col_val in col_vals:
            split_data.append(dataset[dataset[column] == col_val])

        return(split_data)

    def find_best_split(dataset):
        best_gain = 0
        best_feature = 0
        st.subheader("Overall Entropy:")
        st.write(entropy(dataset[targetAttr]))
        for feature in features:
            split_data = split(dataset, feature)
            split_labels = [dataframe[targetAttr] for dataframe in split_data]
            st.subheader(feature)
            gain = information_gain(dataset[targetAttr], split_labels)
            st.write("Gain:",gain)
            if gain > best_gain:
                best_gain, best_feature = gain, feature
        st.subheader("Highest Gain:")
        st.write(best_feature, best_gain)
        return best_feature, best_gain

    # new_data = split(data, find_best_split(data)[0]) 
    # for i in new_data:
    #    st.write(i)
    x = df[features]
    y = df[targetAttr] # Target variable
    dataEncoder = preprocessing.LabelEncoder()
    encoded_x_data = x.apply(dataEncoder.fit_transform)
    # st.header("1.Information Gain")
    # "leaves" (aka decision nodes) are where we get final output
    # root node is where the decision tree starts
    # Create Decision Tree classifer object
    decision_tree =DecisionTreeClassifier(criterion = "entropy",
                random_state = 100,max_depth=3, min_samples_leaf=5)
    # Train Decision Tree Classifer
    decision_tree = decision_tree.fit(encoded_x_data, y)
    
    #plot decision tree
    fig, ax = plt.subplots(figsize=(6, 6)) 
    # #figsize value changes the size of plot
    tree.plot_tree(decision_tree,ax=ax,feature_names=features)
    
    st.pyplot(plt)

    # text_representation = export_text(decision_tree,feature_names=columns[1:-1].tolist(),)
    # st.write(text_representation)
    
    
    
    def tree_to_pseudo(tree, feature_names):

        left = tree.tree_.children_left
        right = tree.tree_.children_right
        threshold = tree.tree_.threshold
        features = [feature_names[i] for i in tree.tree_.feature]
        value = tree.tree_.value

        def recurse(left, right, threshold, features, node, depth=0):
            indent =".                          " * depth*10
            if (threshold[node] != -2):
                st.write(indent+"if ( " + features[node] + " <= " + str(threshold[node]) + " ) {")
                if left[node] != -1:
                    recurse (left, right, threshold, features, left[node], depth+1)
                    st.write(indent+"} else {")
                    if right[node] != -1:
                        recurse (left, right, threshold, features, right[node], depth+1)
                    st.write(indent+"}")
            else:
                st.write(indent+"return " + str(value[node]))

        recurse(left, right, threshold, features, 0)

    tree_to_pseudo(decision_tree, list(features))
