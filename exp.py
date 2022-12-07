# importing packages
from operator import le, length_hint
import operator
import requests
import lxml
from bs4 import BeautifulSoup
from re import template
from urllib.request import urlparse
from urllib.request import urljoin

from scipy.cluster.hierarchy import dendrogram, linkage
import pylab as pl
from sklearn.datasets import load_iris
from random import randint
from sklearn.datasets import make_blobs
from itertools import combinations
from collections import Counter

import streamlit as st
import pandas as pd 
import math
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.metrics import confusion_matrix 
from numpy import *
import numpy as np
from joblib.numpy_pickle_utils import xrange
import itertools

from sklearn import linear_model

from sklearn.linear_model import LinearRegression


from sklearn.linear_model import LogisticRegression



np.set_printoptions(threshold=np.inf)
  

from collections import Counter
from sklearn import tree, preprocessing

from csv import reader
from random import seed
from random import randrange
from math import sqrt
from math import exp
from math import pi

# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.tree import plot_tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import export_graphviz
from sklearn.tree import export_graphviz
from sklearn.tree import _tree
import scipy.stats as stats

  
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

#functions

def median(data):
    data.sort()
    length=len(data)
    if length % 2==1 :
        return round((data[(length+1)//2]),3)
    else:
        return round((data[length//2]+data[length//2+1])/2,3)

# fn to calculate mean
def mean(data):
   length=len(data)
   return round(sum(data)/length,3)

# fn to calculate mode
def mode(data):
    frequency = {}
    for i in data:
        frequency.setdefault(i, 0)
        frequency[i]+=1

    most_frequent = max(frequency.values())
    for i, j in frequency.items():
        if j == most_frequent:
            mode = i
    return mode

#fn to calculate variance
def variance(data):
    mean = sum(data) / len(data)
    res = sum((i - mean) ** 2 for i in data) / (len(data)-1)
    return round(res,3)

#fn to calculate std deviation
def stddeviation(data):
    mean = sum(data) / len(data)
    res = sum((i - mean) ** 2 for i in data) / (len(data)-1)
    return round(math.sqrt(res),4)


st.title("Data Analysis ")





rad = st.sidebar.radio("Menu",["Assignment 1","Assignment 2","Assignment 3","Assignment 4","Assignment 5","Assignment 6","Assignment 7","Assignment 8"])

if rad == "Assignment 8":

    dataset = pd.read_csv('web-polblogs.csv')
    def printf(url):
        st.markdown(f'<p style="color:#000;font:lucida;font-size:20px;">{url}</p>', unsafe_allow_html=True)

    operation = st.selectbox("Operation", ["WebCrawler",'PageRank','HITS']) 
    
    input_url = st.text_input("Paste URL here")
    # Set for storing urls with same domain
    links_intern = set()
    depth = st.number_input("Enter depth (less than 5)", value=1 ,max_value=5, min_value=0)
    links_extern = set()


    def level_crawler(input_url):
        temp_urls = set()
        current_url_domain = urlparse(input_url).netloc

        # Creates beautiful soup object to extract html tags
        beautiful_soup_object = BeautifulSoup(
            requests.get(input_url).content, "lxml")

        # Access all anchor tags from input
        # url page and divide them into internal
        # and external categories
        idx=0
        for anchor in beautiful_soup_object.findAll("a"):
            href = anchor.attrs.get("href")
            if(href != "" or href != None):
                href = urljoin(input_url, href)
                href_parsed = urlparse(href)
                href = href_parsed.scheme
                href += "://"
                href += href_parsed.netloc
                href += href_parsed.path
                final_parsed_href = urlparse(href)
                is_valid = bool(final_parsed_href.scheme) and bool(
                    final_parsed_href.netloc)
                if is_valid:
                    if current_url_domain not in href and href not in links_extern:
                        idx+=1
                        st.write(f"link {idx} - {href}")
                        links_extern.add(href)
                    if current_url_domain in href and href not in links_intern:
                        idx+=1
                        st.write(f"link {idx} - {href}")
                        links_intern.add(href)
                        temp_urls.add(href)
        return temp_urls

    def crawl(input_url, depth):

        if(depth == 0):
            st.write("Page - {}".format(input_url))

        elif(depth == 1):
            level_crawler(input_url)

        else:
            # We have used a BFS approach
            # considering the structure as
            # a tree. It uses a queue based
            # approach to traverse
            # links upto a particular depth.
            queue = []
            queue.append(input_url)
            for j in range(depth):
                st.subheader(f"Level {j} -")
                idx=0
                for count in range(len(queue)):
                    idx+=1
                    url = queue.pop(0)
                    printf(f"Page {idx} : {url} ")
                    urls = level_crawler(url)
                    for i in urls:
                        queue.append(i)

    if st.button("Crawl"):
        crawl(input_url, depth)

    if operation == "PageRank":
        st.dataframe(dataset.head(1000), width=1000, height=500)
        
        # Adjacency Matrix representation in Python


        class Graph(object):

            # Initialize the matrix
            def __init__(self, size):
                self.adjMatrix = []
                self.inbound = dict()
                self.outbound = dict()
                self.pagerank = dict()
                self.vertex = set()
                self.cnt = 0
                # for i in range(size+1):
                #     self.adjMatrix.append([0 for i in range(size+1)])
                self.size = size

            # Add edges
            def add_edge(self, v1, v2):
                if v1 == v2:
                    printf("Same vertex %d and %d" % (v1, v2))
                # self.adjMatrix[v1][v2] = 1
                self.vertex.add(v1)
                self.vertex.add(v2)
                if self.inbound.get(v2,-1) == -1:
                    self.inbound[v2] = [v1]
                else:
                    self.inbound[v2].append(v1)
                if self.outbound.get(v1,-1) == -1:
                    self.outbound[v1] = [v2]
                else:
                    self.outbound[v1].append(v2)

                
                # self.adjMatrix[v2][v1] = 1

            # Remove edges
            # def remove_edge(self, v1, v2):
            #     if self.adjMatrix[v1][v2] == 0:
            #         print("No edge between %d and %d" % (v1, v2))
            #         return
            #     self.adjMatrix[v1][v2] = 0
            #     self.adjMatrix[v2][v1] = 0

            def __len__(self):
                return self.size

            # Print the matrix
            def print_matrix(self):
                # if self.size < 1000:
                #     for row in self.adjMatrix:
                #         for val in row:
                #             printf('{:4}'.format(val), end="")
                #         printf("\n")
                #     printf("Inbound:")
                #     st.write(self.inbound)

                #     printf("Outbound:")
                #     st.write(self.outbound)
                # else:
                pass
            
            def pageRank(self):
                self.cnt = 0
                if len(self.pagerank) == 0:
                    for i in self.vertex:
                        self.pagerank[i] = 1/self.size
                prevrank = self.pagerank
                # print(self.pagerank)
                for i in self.vertex:
                    pagesum = 0.0
                    inb = self.inbound.get(i,-1)
                    if inb == -1:
                        continue
                    for j in inb:
                        pagesum += (self.pagerank[j]/len(self.outbound[j]))
                    self.pagerank[i] = pagesum
                    if (prevrank[i]-self.pagerank[i]) <= 0.1:
                        self.cnt+=1
            def printRank(self):
                printf(self.pagerank)
            def arrangeRank(self):
                sorted_rank = dict( sorted(self.pagerank.items(), key=operator.itemgetter(1),reverse=True))
                # printf(sorted_rank)
                printf("PageRank Sorted : "+str(len(sorted_rank)))
                i = 1
                printf(f"Rank ___ Node ________ PageRank Score")
                for key, rank in sorted_rank.items():
                    if i == 11:
                        break
                    printf(f"{i} _____ {key} ________ {rank}")
                    i += 1

                # st.dataframe(sorted_rank)

        def main():
            g = Graph(7)
            input_list = []
            
            d = 0.5
            for i in range(len(dataset)):
                    input_list.append([dataset.loc[i, 'fromNode'],dataset.loc[i, 'toNode']])
                    g.add_edge(dataset.loc[i, 'fromNode'],dataset.loc[i, 'toNode'])
            size = len(g.vertex)
            if size <= 10000:
                adj_matrix = np.zeros([size+1,size+1])

                for i in input_list:
                    adj_matrix[i[0]][i[1]] = 1

                st.subheader("Adjecency Matrix")
                st.dataframe(adj_matrix, width=1000, height=500)
        
                
            printf("Total Node:"+str(len(g.vertex)))
            printf("Total Edges: "+str(len(input_list)))
            # for i in input_list:

            # g.print_matrix()

            i = 0
            while i<5:
                if g.cnt == g.size:
                    break
                g.pageRank()
                i += 1
            # g.printRank()
            g.arrangeRank()

        main()

    if operation == "HITS":
        input_list = []
        
        st.subheader("Dataset")
        st.dataframe(dataset.head(1000), width=1000, height=500)
        vertex = set()
        for i in range(len(dataset)):
                input_list.append([dataset.loc[i, 'fromNode'],dataset.loc[i, 'toNode']])
                vertex.add(dataset.loc[i, 'fromNode'])
                vertex.add(dataset.loc[i, 'toNode'])
        size = len(vertex)
        adj_matrix = np.zeros([size+1,size+1])

        for i in input_list:
            adj_matrix[i[0]][i[1]] = 1
        
        printf("No of Nodes: "+str(size))
        printf("No of Edges: "+str(len(dataset)))
        st.subheader("Adjecency Matrix")
        st.dataframe(adj_matrix, width=1000, height=500)
        A = adj_matrix
        # st.dataframe(A)
        At = adj_matrix.transpose()
        st.subheader("Transpose of Adj matrix")
        st.dataframe(At)

        u = [1 for i in range(size+1)]
        v = np.matrix([])
        for i in range(5):
            v = np.dot(At,u)
            u = np.dot(A,v)

        # u.sort(reverse=True)
        hubdict = dict()
        for i in range(len(u)):
            hubdict[i]= u[i]
        
        authdict = dict()
        for i in range(len(v)):
            authdict[i]=v[i]

        printf("Hub weight matrix (U)")
        st.dataframe(u)
        printf("Hub weight vector (V)")
        st.dataframe(v)
        hubdict = dict( sorted(hubdict.items(), key=operator.itemgetter(1),reverse=True))
        authdict = dict( sorted(authdict.items(), key=operator.itemgetter(1),reverse=True))
        # printf(sorted_rank)
        printf("HubPages : ")
        i = 1
        printf(f"Rank ___ Node ________ Hubs score")
        for key, rank in hubdict.items():
            if i == 11:
                break
            printf(f"{i} _____ {key} ________ {rank}")
            i += 1

        printf("Authoritative Pages : ")
        i = 1
        printf(f"Rank ___ Node ________ Auth score")
        for key, rank in authdict.items():
            if i == 11:
                break
            printf(f"{i} _____ {key} ________ {rank}")
            i += 1


    
        
    


uploaded_file = st.file_uploader("Choose a file")
st.set_option('deprecation.showPyplotGlobalUse', False)

if uploaded_file :
    df  = pd.read_csv(uploaded_file)

    #################################################################################################################
    if rad == "Assignment 1":
        
    
        st.write(pd.DataFrame(df))
        st.text("")
        #  st.text(df[''])
        
        
    
        colums=df.columns
        attribute= st.selectbox("Select attribute",colums)

        
        


        # Measures of central tendency
        st.header("Measures of central tendency ")
        st.text("")

        col1, col2, col3= st.columns(3)
        data=df[attribute].to_list()
        rads = st.radio("measures of central tendancy",["Mean","Median","Mode"])
        if rads == "Mean":
            st.subheader('Mean')
            st.write( mean(data))
            
        if rads == "Median":
            st.subheader('Median')
            st.write(median(data))
        if rads == "Mode":
            st.subheader('Mode')
            st.write(mode(data))
        st.text("")
        
        col1, col2, col3= st.columns(3)
        rads1 = st.radio("Select ",["Midrange","Variance","Standard deviation"])
        if rads1 == "Midrange":
            st.subheader("Mid Range (max+min)/2")
            st.write(round((max(data)+min(data))/2,3))
        if rads1 == "Variance":
            st.subheader('Variance')
            st.write(variance(data))
        if rads1 == "Standard deviation":
            st.subheader('standard deviation')
            st.write(stddeviation(data))
        st.text("")
    
        st.header("Dispersion of data")
        
        length=len(df)
        data=df[attribute].to_list()

        
        
        st.subheader("Range (max-min)")
        st.write(round(max(data)-min(data),3))
        
        st.subheader("Quartile (Q1)")
        Q1=median(data[0:length//2])
        st.write(round(Q1,3))
        
        st.subheader("Quartile (Q2)")
        Q2=median(data)
        st.write(round(Q2,3))
        st.text("")
        
        
        
        st.subheader("Quartile (Q3)")
        Q3=median(data[length//2:])
        st.write(round(Q3,3))
        
        st.subheader("Interquartile range (Q3-Q1)")
        Q1=median(data[0:length//2])
        Q3=median(data[length//2:])
        st.write(round(Q3-Q1,3))
        st.text("")
        
        st.subheader("Five Number Summary")
        col1, col2, col3 ,col4,col5 = st.columns(5)
        data.sort()
        with col1:
            st.text('Min')
            st.write(min(data))
        with col2:
            st.text('Q1')
            st.write(median(data[0:length//2]))
        with col3:
            st.text('Median')
            st.write(median(data))
        with col4:
            st.text('Q2')
            st.write(median(data[length//2:]))
        with col5:
            st.text('max')
            st.write(max(data))
        st.text("")





        st.header('Graphical Representation:')

        plt.rcParams['figure.figsize'] = [8, 4]
        st.write("Histogram")
        fig, ax = plt.subplots()
        plt.locator_params(nbins = 15)
        plt.xlabel(attribute)
        plt.ylabel("count")
        ax.hist(data)
        st.pyplot(fig)

        plt.clf()
        
        st.text("")
        st.text("")
        st.text("")

        xlabel = st.selectbox("xLabel",df.columns)
        ylabel = st.selectbox("yLabel",df.columns)
        plt.locator_params(nbins = 10)
        plt.scatter(df[xlabel],df[ylabel], c ="black", s=5)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        st.write("Scatter Plot")
        st.pyplot(plt)
        plt.clf()

        indices = []
        ind=0
        for i in df.columns:
            if df.dtypes[i]!=object:
                indices.append(ind)
            ind+=1
        st.text("")
        st.text("")
        st.text("")
        st.write("Box Plot")
        data= df.iloc[:,indices].values
        fig = plt.figure(figsize =(10, 7))
        plt.boxplot(data)
        st.pyplot(plt)
    

    ###########################################################################################################
    if rad == "Assignment 2":

        ####chi square test#######################################################################
        st.write("Chi square test")

        xlabel = st.selectbox("First attribute",df.columns)
        ylabel = st.selectbox("Second attribute",df.columns)
        plt.locator_params(nbins = 10)

        print(xlabel)
        contigency_table = pd.crosstab(df[xlabel],df[ylabel],margins=True,margins_name="All")
        st.text(contigency_table)
        rows = df[xlabel].unique()
        columns = df[ylabel].unique()
        chi_square = 0.0
        for i in columns:
            for j in rows:
                obs = contigency_table[i][j]
                expected = (contigency_table[i]['All'] * contigency_table['All'][j])/(contigency_table['All']['All'])
                chi_square = chi_square + ((obs - expected)**2/expected)
        p_value = 1 - stats.chi2.cdf(chi_square,(len(rows) - 1)*(len(columns) - 1))
        dof = (len(columns) - 1)*(len(rows) - 1)

        st.subheader("chi-square value")
        st.write(chi_square)
        st.subheader("degree of freedom")
        st.write(dof)
        if(dof<chi_square):
            st.subheader("columns have the corealation")
        else:
            st.subheader("columns don't have any relation")


        ####correlation coeficient#####################################################
        sum = 0
        for i in range(len(df)):
            sum += df.loc[i, xlabel]
        avg1 = sum/len(df)
        sum = 0
        for i in range(len(df)):
            sum += (df.loc[i, xlabel]-avg1)*(df.loc[i, xlabel]-avg1)
        var1 = sum/(len(df))
        sd1 = math.sqrt(var1)
        sum = 0
        for i in range(len(df)):
            sum += df.loc[i, ylabel]
        avg2 = sum/len(df)
        sum = 0
        for i in range(len(df)):
            sum += (df.loc[i, ylabel]-avg2)*(df.loc[i, ylabel]-avg2)
        var2 = sum/(len(df))
        sd2 = math.sqrt(var2)

        st.write("First column's average: ")
        st.write(round(avg1,3))

        st.write("First column's std deviation:")
        st.write(round(sd1,3))

        st.write("Second column's average: ")
        st.write(round(avg2,3))

        st.write("Second column's std deviation:")
        st.write(round(sd2,3))

        sum = 0
        for i in range(len(df)):
            sum += (df.loc[i, xlabel]-avg1)*(df.loc[i, ylabel]-avg2)
        covariance = sum/len(df)
        pearsonCoeff = covariance/(sd1*sd2) 

        st.write("Pearson coeffiecient:")
        st.write(pearsonCoeff)

        if pearsonCoeff>0:
            st.write("Attributes are positively correlated")
        elif pearsonCoeff<0:
            st.write("Attributes are negatively correlated")
        elif pearsonCoeff == 0:
            st.write("There is no relation at all")
        
        ####Normalization techniques
        st.header("Normalization techniques")

        rad1 = st.radio("Choose technique for normalization",["Min-Max","Z-Score","Decimal-Scaling"])
        attribute_to_be_normalized = st.selectbox("Select column",df.columns)

        ####min-max normalization
        if rad1 == "Min-Max":
            min = df[attribute_to_be_normalized].min()
            max = df[attribute_to_be_normalized].max()
            for i in range(len(df)):
                df.loc[i, attribute_to_be_normalized] = ((df.loc[i, attribute_to_be_normalized]-min)/(max-min))

        ####z-score normalization
        if rad1 == "Z-Score":
            sum = 0
            for i in range(len(df)):
                sum += df.loc[i, attribute_to_be_normalized]
            avg = sum/len(df)
            sum = 0
            for i in range(len(df)):
                sum += (df.loc[i, attribute_to_be_normalized]-avg)*(df.loc[i, attribute_to_be_normalized]-avg)
            var = sum/(len(df))
            sd = math.sqrt(var2)

            for i in range(len(df)):
                df.loc[i, attribute_to_be_normalized] = ((df.loc[i, attribute_to_be_normalized]-avg1)/sd1)

        if rad1 == "Decimal-Scaling":
            j=0
            max = df[attribute_to_be_normalized].max()
            while max > 1:
                max /= 10
                j += 1
            for i in range(len(df)):
                df.loc[i, attribute_to_be_normalized] = ((df.loc[i, attribute_to_be_normalized])/(pow(10,j)))
        
        st.dataframe(df[attribute_to_be_normalized])


    ###########################################################################################################
    if rad == "Assignment 3" or rad == "Assignment 4":

        colums=df.columns

        targetAttr=st.selectbox("Choose Target Attribute",colums)       
        st.header("Decision Tree")
        data=df
        features = list(colums)
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

        new_data = split(data, find_best_split(data)[0]) 
        # for i in new_data:
        #    st.write(i)

        features = list(colums)
        features.remove(targetAttr)
        x = df[features]
        y = df[targetAttr] # Target variable

        dataEncoder = preprocessing.LabelEncoder()
        encoded_x_data = x.apply(dataEncoder.fit_transform)

        st.header("1.Information Gain")
        # "leaves" (aka decision nodes) are where we get final output
        # root node is where the decision tree starts
        # Create Decision Tree classifer object
        decision_tree = DecisionTreeClassifier(criterion="entropy")
        # Train Decision Tree Classifer
        decision_tree = decision_tree.fit(encoded_x_data, y)
        
        #plot decision tree
        fig, ax = plt.subplots(figsize=(6, 6)) 
        #figsize value changes the size of plot
        tree.plot_tree(decision_tree,ax=ax,feature_names=features)
        
        st.pyplot(plt)

        st.header("2.Gini Index")
        decision_tree = DecisionTreeClassifier(criterion="gini")
        # Train Decision Tree Classifer
        decision_tree = decision_tree.fit(encoded_x_data, y)
        
        fig, ax = plt.subplots(figsize=(6, 6)) 
        tree.plot_tree(decision_tree,ax=ax,feature_names=features)
        plt.show()
        st.pyplot(plt)

        X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=1)

        # Create Decision Tree classifer object
        clf = DecisionTreeClassifier(max_depth=2, random_state=1)

        # Train Decision Tree Classifer
        clf = clf.fit(X_train, y_train)

        # Predict the response for test dataset
        y_pred = clf.predict(X_test)

        c_matrix = confusion_matrix(y_test, y_pred)

        tp = c_matrix[1][1]
        tn = c_matrix[2][2]
        fp = c_matrix[1][2]
        fn = c_matrix[2][1]


        st.subheader("confusion Matrix:")
        st.write(c_matrix)

        # Tabulate the results in confusion matrix and evaluate the performance of above classifier using following metrics :
        st.write('Tabulate the results in confusion matrix and evaluate the performance of above classifier using following metrics :')

        
        st.write("Model Accuracy: " + str(metrics.accuracy_score(y_test, y_pred)))
        # precision score
        val = metrics.precision_score(y_test, y_pred, average='macro')
        print('Precision score : ' + str(val))
        st.write('Precision score : ' + str(val))


        # Accuracy score
        val = metrics.accuracy_score(y_test, y_pred)
        st.write('Accuracy score : ' + str(val))

        #Assignment 4
        st.header("Rule Base Classifier")
        # get the text representation
        text_representation = tree.export_text(clf,feature_names=features)
        st.text(text_representation)

        #Extract Code Rules
        st.subheader("Extract Code Rules")



        def tree_to_code(tree, feature_names):
            tree_ = tree.tree_
            feature_name = [
                feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
                for i in tree_.feature
            ]
            feature_names = [f.replace(" ", "_")[:-5] for f in feature_names]
            st.text("def predict({}):".format(", ".join(feature_names)))

            def recurse(node, depth):
                indent = "    " * depth
                if tree_.feature[node] != _tree.TREE_UNDEFINED:
                    name = feature_name[node]
                    threshold = tree_.threshold[node]
                    st.text("{}if {} <= {}:".format(indent, name, np.round(threshold,2)))
                    recurse(tree_.children_left[node], depth + 1)
                    st.text("{}else:  # if {} > {}".format(indent, name, np.round(threshold,2)))
                    recurse(tree_.children_right[node], depth + 1)
                else:
                    st.text("{}return {}".format(indent, tree_.value[node]))

            recurse(0, 1)
        
        tree_to_code(decision_tree,features)

        radk = st.selectbox("Check",["Coverage","Accuracy","Toughness"])

        
        

    if rad == "Assignment 5":

        st.dataframe(df)


        rad5 = st.radio("Select",["Regression classifier","Naive Bayesian classifier","k-NN classifier","ANN classifier"])
        if rad5 =="Regression classifier":
            colums=df.columns
            features=list(colums)
            features.remove(features[0])
            # find categorical variables
            categorical = [var for var in df.columns if df[var].dtype=='O']
            targetAttr=st.selectbox("Choose Target Attribute",categorical) 
            st.text("Categorial Attributes:")
            st.text(categorical)
            numerical = [var for var in df.columns if df[var].dtype!='O']
            st.text("Numeric Attributes:")
            st.text(numerical)

            X = df.drop([targetAttr], axis=1)
            y = df[targetAttr]

            # st.write(X)
            # st.write

            def sigmoid(X, weight):
                z = np.dot(X, weight)
                return 1 / (1 + np.exp(-z))

            def loss(h, y):
                return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

            def gradient_descent(X, h, y):
                return np.dot(X.T, (h - y)) / y.shape[0]
                
            def update_weight_loss(weight, learning_rate, gradient):
                return weight - learning_rate * gradient
            
            # train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.7, random_state = 0)

            st.write("Train Test Shape:")
            st.write(X_train.shape, X_test.shape)

            #feature Scaling
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            # Fitting Logistic Regression to the Training set
            classifier =  LogisticRegression(random_state = 0, solver='lbfgs', multi_class='auto')
            classifier.fit(X_train, y_train)

            # Predicting the Test set results
            y_pred = classifier.predict(X_test)
            # Predict probabilities
            probs_y=classifier.predict_proba(X_test)
            ### Print results 
            probs_y = np.round(probs_y, 2)
            
            st.subheader("Test set Result Predication:")
            st.write(y_pred)
            st.subheader("Corresponding %")
            st.write(probs_y)


            st.write("confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            st.write(cm)

            # Classifiaction Report
            st.subheader("Performance Evaluation")
            st.text(classification_report(y_test, y_pred))

        #################################################################################################
        if rad5 == "Naive Bayesian classifier":

            dataset = df
            x = dataset.iloc[:, [2, 3]].values  
            y = dataset.iloc[:, 4].values  
            
            # Splitting the dataset into the Training set and Test set  
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)  
            
            # Feature Scaling  
            sc = StandardScaler()  
            x_train = sc.fit_transform(x_train)  
            x_test = sc.transform(x_test)
            
            from sklearn.naive_bayes import GaussianNB 
            classifier = GaussianNB()  
            classifier.fit(x_train, y_train) 
            y_pred = classifier.predict(x_test)
            st.subheader("Test set Result Predication:")
            #ds = np.array([x_test,y_pred])
            st.write(y_pred)
            cm = confusion_matrix(y_test, y_pred)  
            st.dataframe(cm)
            st.subheader("Performance Evaluation")
            st.text(classification_report(y_test, y_pred))
        #################################################################################
        if rad5 == "k-NN classifier":

            arr = []
            arr = df["Species"].unique()

            x= df.iloc[:, [0,1,2,3]].values  
            y= df.iloc[:, 4].values

            # Splitting the dataset into training and test set.  
            
            x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)  
            
            #feature Scaling  
            
            st_x= StandardScaler()    
            x_train= st_x.fit_transform(x_train)    
            x_test= st_x.transform(x_test)  

            print(y_train)
            print("train")
            print(y_test)
            input = []
            input.append(st.number_input("SepalLength", min_value=None, max_value=None, step=None))
            input.append(st.number_input("SepalWidth", min_value=None, max_value=None, step=None))
            input.append(st.number_input("PetalLength", min_value=None, max_value=None, step=None))
            input.append(st.number_input("PetalWidth", min_value=None, max_value=None, step=None))
            point = input
            distance_points = []
            print(np.linalg.norm(point - x_train[2]))
            j=0
            for i in range(len(x_train)):
                # distance_points[i] = np.linalg.norm(point - x_train[j])
                temp = point - x_train[i]
                sum = np.dot(temp.T,temp)
                distance_points.append(np.sqrt(sum))

            for i in range(len(x_train),len(df)):
                distance_points.append(1000)

            classifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
            classifier.fit(x_train, y_train)  

            y_pred= classifier.predict(x_test)  

            m= confusion_matrix(y_test, y_pred)  
            
            # print(point)
            # print(x_train[2])
            
            df["distance"] = distance_points
            
            x= df.iloc[:, [0,1,2,3,5]].values  
            y= df.iloc[:, 4].values
           
            x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25)  
            
             
            
            st_x= StandardScaler()    
            x_train= st_x.fit_transform(x_train)    
            x_test= st_x.transform(x_test)
            
            

            df = df.sort_values(by=['distance'])
            # print(df1)
            # st.subheader("After sorting")
            # st.dataframe(df)

            k_value = st.selectbox("k-value",[1,3,5,7])

            df_first_k = df[1:k_value+1]

            st.dataframe(df_first_k)

            nearest_neighbour = mode(df_first_k['Species'])
            st.subheader("Nearest neighbour is")
            st.write(nearest_neighbour )


        if rad5 == "ANN classifier":

            def Sigmoid(Z):
                return 1/(1+np.exp(-Z))

            def Relu(Z):
                return np.maximum(0,Z)

            def dRelu2(dZ, Z):    
                dZ[Z <= 0] = 0    
                return dZ

            def dRelu(x):
                x[x<=0] = 0
                x[x>0] = 1
                return x

            def dSigmoid(Z):
                s = 1/(1+np.exp(-Z))
                dZ = s * (1-s)
                return dZ

            class dlnet:
                def __init__(self, x, y):
                    self.debug = 0
                    self.X=x
                    self.Y=y
                    self.Yh=np.zeros((1,self.Y.shape[1])) 
                    self.L=2
                    self.dims = [9, 15, 1] 
                    self.param = {}
                    self.ch = {}
                    self.grad = {}
                    self.loss = []
                    self.lr=0.003
                    self.sam = self.Y.shape[1]
                    self.threshold=0.5
                    
                def nInit(self):    
                    np.random.seed(1)
                    self.param['W1'] = np.random.randn(self.dims[1], self.dims[0]) / np.sqrt(self.dims[0]) 
                    self.param['b1'] = np.zeros((self.dims[1], 1))        
                    self.param['W2'] = np.random.randn(self.dims[2], self.dims[1]) / np.sqrt(self.dims[1]) 
                    self.param['b2'] = np.zeros((self.dims[2], 1))                
                    return 

                def forward(self):    
                    Z1 = self.param['W1'].dot(self.X) + self.param['b1'] 
                    A1 = Relu(Z1)
                    self.ch['Z1'],self.ch['A1']=Z1,A1
                    
                    Z2 = self.param['W2'].dot(A1) + self.param['b2']  
                    A2 = Sigmoid(Z2)
                    self.ch['Z2'],self.ch['A2']=Z2,A2

                    self.Yh=A2
                    loss=self.nloss(A2)
                    return self.Yh, loss

                def nloss(self,Yh):
                    loss = (1./self.sam) * (-np.dot(self.Y,np.log(Yh).T) - np.dot(1-self.Y, np.log(1-Yh).T))    
                    return loss

                def backward(self):
                    dLoss_Yh = - (np.divide(self.Y, self.Yh ) - np.divide(1 - self.Y, 1 - self.Yh))    
                    
                    dLoss_Z2 = dLoss_Yh * dSigmoid(self.ch['Z2'])    
                    dLoss_A1 = np.dot(self.param["W2"].T,dLoss_Z2)
                    dLoss_W2 = 1./self.ch['A1'].shape[1] * np.dot(dLoss_Z2,self.ch['A1'].T)
                    dLoss_b2 = 1./self.ch['A1'].shape[1] * np.dot(dLoss_Z2, np.ones([dLoss_Z2.shape[1],1])) 
                                        
                    dLoss_Z1 = dLoss_A1 * dRelu(self.ch['Z1'])        
                    dLoss_A0 = np.dot(self.param["W1"].T,dLoss_Z1)
                    dLoss_W1 = 1./self.X.shape[1] * np.dot(dLoss_Z1,self.X.T)
                    dLoss_b1 = 1./self.X.shape[1] * np.dot(dLoss_Z1, np.ones([dLoss_Z1.shape[1],1]))  
                    
                    self.param["W1"] = self.param["W1"] - self.lr * dLoss_W1
                    self.param["b1"] = self.param["b1"] - self.lr * dLoss_b1
                    self.param["W2"] = self.param["W2"] - self.lr * dLoss_W2
                    self.param["b2"] = self.param["b2"] - self.lr * dLoss_b2
                    
                    return


                def pred(self,x, y):  
                    self.X=x
                    self.Y=y
                    comp = np.zeros((1,x.shape[1]))
                    pred, loss= self.forward()    
                
                    for i in range(0, pred.shape[1]):
                        if pred[0,i] > self.threshold: comp[0,i] = 1
                        else: comp[0,i] = 0
                
                    print("Acc: " + str(np.sum((comp == y)/x.shape[1])))
                    
                    return comp
                
                def gd(self,X, Y, iter = 3000):
                    np.random.seed(1)                         
                
                    self.nInit()
                
                    for i in range(0, iter):
                        Yh, loss=self.forward()
                        self.backward()
                    
                        if i % 500 == 0:
                            print ("Cost after iteration %i: %f" %(i, loss))
                            self.loss.append(loss)

                    plt.plot(np.squeeze(self.loss))
                    plt.ylabel('Loss')
                    plt.xlabel('Iter')
                    plt.title("Lr =" + str(self.lr))
                    st.pyplot(plt)
                
                    return 
            
            def plotCf(a,b,t):
                cf =confusion_matrix(a,b)
                st.dataframe(cf)
                plt.imshow(cf,cmap=plt.cm.Blues,interpolation='nearest')
                plt.colorbar()
                plt.title(t)
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                tick_marks = np.arange(len(set(a))) # length of classes
                class_labels = ['0','1']
                plt.xticks(tick_marks,class_labels)
                plt.yticks(tick_marks,class_labels)
                thresh = cf.max() / 2.
                for i,j in itertools.product(range(cf.shape[0]),range(cf.shape[1])):
                    plt.text(j,i,format(cf[i,j],'d'),horizontalalignment='center',color='white' if cf[i,j] >thresh else 'black')
                
                st.pyplot(plt)

                

                # print(type(data))
            df = pd.read_csv("C:/Users/91721/Downloads/breast-cancer-wisconsin1.csv",header=None)
            df = df[~df[6].isin(['?'])]
            df = df.astype(float)
            df.iloc[:,10].replace(2, 0,inplace=True)
            df.iloc[:,10].replace(4, 1,inplace=True)

            df.head(3)
            scaled_df=df
            names = df.columns[0:10]
            scaler = MinMaxScaler() 
            scaled_df = scaler.fit_transform(df.iloc[:,0:10]) 
            scaled_df = pd.DataFrame(scaled_df, columns=names)
            x=scaled_df.iloc[0:500,1:10].values.transpose()
            y=df.iloc[0:500,10:].values.transpose()

            xval=scaled_df.iloc[501:683,1:10].values.transpose()
            yval=df.iloc[501:683,10:].values.transpose()

            print(df.shape, x.shape, y.shape, xval.shape, yval.shape)

            nn = dlnet(x,y)
            nn.lr=0.07
            nn.dims = [9, 15, 1]
            nn.gd(x, y, iter = 67000)
            pred_train = nn.pred(x, y)
            pred_test = nn.pred(xval, yval)
            print("Pred test is:",pred_test)
            st.write("Accuracy:",str(np.sum((pred_test == yval)/xval.shape[1])))
            nn.threshold=0.5

            nn.X,nn.Y=x, y 
            target=np.around(np.squeeze(y), decimals=0).astype(np.int)
            predicted=np.around(np.squeeze(nn.pred(x,y)), decimals=0).astype(np.int)
            plotCf(target,predicted,'Cf Training Set')

            nn.X,nn.Y=xval, yval 
            target=np.around(np.squeeze(yval), decimals=0).astype(np.int)
            predicted=np.around(np.squeeze(nn.pred(xval,yval)), decimals=0).astype(np.int)
            plotCf(target,predicted,'Cf Validation Set')
            nn.X,nn.Y=xval, yval 
            yvalh, loss = nn.forward()
            print("\ny",np.around(yval[:,0:50,], decimals=0).astype(np.int))       
            print("\nyh",np.around(yvalh[:,0:50,], decimals=0).astype(np.int),"\n")


    if rad == "Assignment 9":

        # x= df.iloc[:, [0,1,2,3]].values  
        # y= df.iloc[:, 4].values

        # x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0)

        # distance_matrix = [[]]

        length = len(df.axes[0])
        
        # k = 0

        # for i in range (0,length):
            
        #     k = 0
        #     for j in range(0,length):
        #         temp = x_train[i]-x_train[j]
        #         sum = np.dot(temp.T,temp)
        #         distance_matrix.append(np.sqrt(sum))
        #     k = k+1
        
        # st.write(distance_matrix)


        def euclideanDistance(data_1, data_2, data_len):
            dist = 0
            for i in range(data_len):
                dist = dist + np.square(data_1[i] - data_2[i])
            return np.sqrt(dist)

        distance_matrix = [[0 for i in range(length)] for i in range(length)]
        k=0
        for it in df.index:
            
            point1 = []
            point1.append(df['SepalLengthCm'][it])
            point1.append(df['SepalWidthCm'][it])
            point1.append(df['PetalLengthCm'][it])
            point1.append(df['PetalWidhtCm'][it])
            # print(point1)
            tmp = []
            for it1 in df.index:
                point2 = []
                point2.append(df['SepalLengthCm'][it1])
                point2.append(df['SepalWidthCm'][it1])
                point2.append(df['PetalLengthCm'][it1])
                point2.append(df['PetalWidhtCm'][it1])
                # print(point2)
                dt = euclideanDistance(point1, point2, 4)
                
                # print(dt)
                distance_matrix[it][it1] = dt
            k = k+1
        

        st.dataframe(distance_matrix)

    if rad == "Assignment 6":


        ####AGNES
        length = len(df.axes[0])
        
        def euclideanDistance(data_1, data_2, data_len):
            dist = 0
            for i in range(data_len):
                dist = dist + np.square(data_1[i] - data_2[i])
            return np.sqrt(dist)

        distance_matrix = [[0 for i in range(length)] for i in range(length)]
        k=0
        for it in df.index:
            
            point1 = []
            point1.append(df['SepalLengthCm'][it])
            point1.append(df['SepalWidthCm'][it])
            point1.append(df['PetalLengthCm'][it])
            point1.append(df['PetalWidhtCm'][it])
            # print(point1)
            tmp = []
            for it1 in df.index:
                point2 = []
                point2.append(df['SepalLengthCm'][it1])
                point2.append(df['SepalWidthCm'][it1])
                point2.append(df['PetalLengthCm'][it1])
                point2.append(df['PetalWidhtCm'][it1])
                # print(point2)
                dt = euclideanDistance(point1, point2, 4)
                
                # print(dt)
                distance_matrix[it][it1] = dt
            k = k+1
        

        st.dataframe(distance_matrix)

    
        cols = []
        for i in df.columns[:-1]:
            cols.append(i)
        
        attribute1 = st.selectbox("Select Attribute 1", cols)
        attribute2 = st.selectbox("Select Attribute 2", cols)
        dataset = []
        arr1 = []
        arr2 = []
        for i in range(len(df)):
            arr1.append(df.loc[i, attribute1])
        
        for i in range(len(df)):
            arr2.append(df.loc[i, attribute2])
        
        for i in range(len(arr1)):
            tmp = []
            tmp.append(arr1[i])
            tmp.append(arr2[i])
            dataset.append(tmp)

        def dist(a, b):
            return math.sqrt(math.pow(a[0]-b[0], 2)+math.pow(a[1]-b[1], 2))


        def dist_min(Ci, Cj):
            return min(dist(i, j) for i in Ci for j in Cj)


        def dist_max(Ci, Cj):
            return max(dist(i, j) for i in Ci for j in Cj)
    

        def dist_avg(Ci, Cj):
            return sum(dist(i, j) for i in Ci for j in Cj)/(len(Ci)*len(Cj))

        def find_Min(M):
            min = 1000
            x = 0
            y = 0
            for i in range(len(M)):
                for j in range(len(M[i])):
                    if i != j and M[i][j] < min:
                        min = M[i][j]
                        x = i
                        y = j
            return (x, y, min)

        def AGNES(dataset, dist, k):
            C = []
            M = []
            for i in dataset:
                Ci = []
                Ci.append(i)
                C.append(Ci)
   
            for i in C:
                Mi = []
                for j in C:
                    
                    Mi.append(dist(i, j))
                M.append(Mi)
  
            q = len(dataset)
     
            while q > k:
                x, y, min = find_Min(M)
                C[x].extend(C[y])
                C.remove(C[y])
                M = []
                for i in C:
                    Mi = []
                    for j in C:
                        Mi.append(dist(i, j))
                    M.append(Mi)
                q -= 1
            # st.write(M)
            return C

        def draw(C):
            st.write("Plot of cluster using AGNES")
            colValue = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
            c = ["Setosa", "Versicolor", "Virginica"]
            for i in range(len(C)):
                coo_X = []
                coo_Y = []
                for j in range(len(C[i])):
                    coo_X.append(C[i][j][0])
                    coo_Y.append(C[i][j][1])
                pl.xlabel(attribute1)
                pl.ylabel(attribute2)
                pl.scatter(
                    coo_X, coo_Y, color=colValue[i % len(colValue)], label=i)

            pl.legend(loc='upper right')
            st.pyplot()
        n = st.number_input('Insert value for K', step=1, min_value=1)

        C = AGNES(dataset, dist_avg, n)
        draw(C)
        st.write("Dendogram plot")
        iris = load_iris()
        dist_sin = linkage(iris.data, method="ward")
        plt.figure(figsize=(20, 15))
        dendrogram(dist_sin, above_threshold_color='#070dde',orientation='top', leaf_rotation=90)
        plt.xlabel('Index')
        plt.ylabel('Distance')
        plt.title("Dendrogram", fontsize=18)
        plt.show()
        st.pyplot()


        ######DIANA
        arr = []
        for i in range(len(df)):
            arr.append([df.loc[i, attribute1], df.loc[i, attribute2]])
        n = len(arr)
        k = int(st.number_input("Enter no of Clusters (k): ", min_value=1, step=1))

        minPoints = 0
        if len(arr) % k == 0:
            minPoints = len(arr)//k
        else:
            minPoints = (len(arr)//k)+1
        # print(len(arr))
        print(minPoints)

        def Euclid(a, b):
            sum_sq = np.sum(np.square(a - b))
            return np.sqrt(sum_sq)

        points = [[0]]

        def findPoints(point):
            max = 0
            pt = -1
            for i in point:
                for j in range(len(arr)):
                    if j in point:
                        continue
                    else:
                        dis = np.sqrt(np.sum(np.square(arr[i]) - np.array(arr[j])))
                        if max < dis:
                            max = dis
                            # print(max)
                            pt = j
            return pt

        travetsedPoints = [0]
        for i in range(0, k):
            if len(travetsedPoints) >= len(arr):
                break

           

            while (len(points[i]) < minPoints):
                # while(True):
                pt = findPoints(travetsedPoints)
                if pt in travetsedPoints:
                    break
                travetsedPoints.append(pt)
                points[i].append(pt)
            points.append([])
        points.remove([])
        
        colarr = []

        for i in range(k):
            colarr.append('#%06X' % randint(0, 0xFFFFFF))

        i = 0
        cluster = []
        for j in range(k):
            cluster.append(j)


        j = 0

        def findIndex(ptarr):
            # print("Ptarr: ", ptarr)
            for j in range(len(points)):
                if ptarr in points[j]:
                    return j

        fig, axes = plt.subplots(1, figsize=(10, 7))
        clusters = []
        for i in range(k):
            clusters.append([[], []])

        for i in range(len(arr)):
            j = findIndex(i)
            clusters[j % k][0].append(arr[i][0])
            clusters[j % k][1].append(arr[i][1])

            # print(i)
            # plt.scatter(arr[i][0],arr[i][1], color = colarr[j])
        for i in range(len(clusters)):
            plt.scatter(clusters[i][0], clusters[i][1],
                        color=colarr[i % k], label=cluster[i])
        plt.title("Cluster plot using DIANA")
        plt.xlabel(attribute2)
        plt.ylabel(attribute1)
        plt.legend(loc=1, prop={'size': 15})

        
        st.pyplot()

        
        st.write("Dendogram plot")
        dist_sin = linkage(iris.data, method="ward")
        plt.figure(figsize=(20, 15))
        dendrogram(dist_sin, above_threshold_color='#070dde',orientation='top', leaf_rotation=90)
        plt.xlabel('Index')
        plt.ylabel('Distance')
        plt.title("Dendrogram", fontsize=18)
        plt.show()
        st.pyplot()


        ####DBSCAN

        def calDist(X1, X2):
            sum = 0
            for x1, x2 in zip(X1, X2):
                sum += (x1 - x2) ** 2
            return sum ** 0.5

        def getNeighbour(data, dataSet, e):
            res = []
            for i in range(len(dataSet)):
                if calDist(data, dataSet[i]) < e:
                    res.append(i)
            return res

        def DBSCAN(dataSet, e, minPts):
            coreObjs = {}
            C = {}
            n = dataset
            for i in range(len(dataSet)):
                neibor = getNeighbour(dataSet[i], dataSet, e)
                if len(neibor) >= minPts:
                    coreObjs[i] = neibor
            oldCoreObjs = coreObjs.copy()
            # st.write(oldCoreObjs)
            # CoreObjs set of COres points
            k = 0
            notAccess = list(range(len(dataset)))

            # his will check the relation of core point with each other
            while len(coreObjs) > 0:
                OldNotAccess = []
                OldNotAccess.extend(notAccess)
                cores = coreObjs.keys()
                randNum = randint(0, len(cores))
                cores = list(cores)
                core = cores[randNum]
                queue = []
                queue.append(core)
                notAccess.remove(core)
                while len(queue) > 0:
                    q = queue[0]
                    del queue[0]
                    if q in oldCoreObjs.keys():
                        delte = [val for val in oldCoreObjs[q]
                                 if val in notAccess]
                        queue.extend(delte)
                        notAccess = [
                            val for val in notAccess if val not in delte]
                k += 1
                C[k] = [val for val in OldNotAccess if val not in notAccess]
                for x in C[k]:
                    if x in coreObjs.keys():
                        del coreObjs[x]
            # st.write(C)
            return C

        def draw(C, dataSet):
            color = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
            vis = set()
            for i in C.keys():
                X = []
                Y = []
                datas = C[i]
                for k in datas:
                    vis.add(k)
                for j in range(len(datas)):
                    X.append(dataSet[datas[j]][0])
                    Y.append(dataSet[datas[j]][1])
                plt.scatter(X, Y, marker='o',
                            color=color[i % len(color)], label=i)
            vis = list(vis)
            unvisited1 = []
            unvisited2 = []
            for i in range(len(dataSet)):
                if i not in vis:
                    unvisited1.append(dataSet[i][0])
                    unvisited2.append(dataSet[i][1])
            st.write("Plot of cluster's after DBSCAN ")
            plt.xlabel(attribute1)
            plt.ylabel(attribute2)
            plt.scatter(unvisited1, unvisited2, marker='o', color='black')
            plt.legend(loc='lower right')
            plt.show()
            st.pyplot()

        cols = []
        for i in df.columns[:-1]:
            cols.append(i)
        # atr1, atr2 = st.columns(2)
        dataset = []
        arr1 = []
        arr2 = []
        for i in range(len(df)):
            arr1.append(df.loc[i, attribute1])
        for i in range(len(df)):
            arr2.append(df.loc[i, attribute2])
        for i in range(len(arr1)):
            tmp = []
            tmp.append(arr1[i])
            tmp.append(arr2[i])
            dataset.append(tmp)
        r = st.number_input('Insert value for eps', value=0.09)
        mnp = st.number_input(
            'Insert mimimum number of points in cluster', step=1, value=7)
        C = DBSCAN(dataset, r, mnp)
        draw(C, dataset)

        ####K-means

        class color:
            PURPLE = '\033[95m'
            CYAN = '\033[96m'
            DARKCYAN = '\033[36m'
            BLUE = '\033[94m'
            GREEN = '\033[92m'
            YELLOW = '\033[93m'
            RED = '\033[91m'
            BOLD = '\033[1m'
            UNDERLNE = '\033[4m'
            END = '\033[0m'

        def plot_data(X):
            plt.figure(figsize=(7.5, 6))
            for i in range(len(X)):
                plt.scatter(X[i][0], X[i][1], color='k')

        def random_centroid(X, k):
            random_idx = [np.random.randint(len(X)) for i in range(k)]
            centroids = []
            for i in random_idx:
                centroids.append(X[i])
            return centroids

        def assign_cluster(X, ini_centroids, k):
            cluster = []
            for i in range(len(X)):
                euc_dist = []
                for j in range(k):
                    euc_dist.append(np.linalg.norm(
                        np.subtract(X[i], ini_centroids[j])))
                idx = np.argmin(euc_dist)
                cluster.append(idx)
            return np.asarray(cluster)

        def compute_centroid(X, clusters, k):
            centroid = []
            for i in range(k):
                temp_arr = []
                for j in range(len(X)):
                    if clusters[j] == i:
                        temp_arr.append(X[j])
                centroid.append(np.mean(temp_arr, axis=0))
            return np.asarray(centroid)

        def difference(prev, nxt):
            diff = 0
            for i in range(len(prev)):
                diff += np.linalg.norm(prev[i]-nxt[i])
            return diff

        def show_clusters(X, clusters, centroids, ini_centroids, mark_centroid=True, show_ini_centroid=True, show_plots=True):
            cols = {0: 'r', 1: 'b', 2: 'g', 3: 'coral', 4: 'c', 5: 'lime'}
            fig, ax = plt.subplots(figsize=(7.5, 6))
            for i in range(len(clusters)):
                ax.scatter(X[i][0], X[i][1], color=cols[clusters[i]])
            for j in range(len(centroids)):
                ax.scatter(centroids[j][0], centroids[j]
                           [1], marker='*', color=cols[j])
                if show_ini_centroid == True:
                    ax.scatter(
                        ini_centroids[j][0], ini_centroids[j][1], marker="+", s=150, color=cols[j])
            if mark_centroid == True:
                for i in range(len(centroids)):
                    ax.add_artist(plt.Circle(
                        (centroids[i][0], centroids[i][1]), 0.4, linewidth=2, fill=False))
                    if show_ini_centroid == True:
                        ax.add_artist(plt.Circle(
                            (ini_centroids[i][0], ini_centroids[i][1]), 0.4, linewidth=2, color='y', fill=False))
            ax.set_xlabel(attribute1)
            ax.set_ylabel(attribute2)
            ax.set_title("K-means Clustering")
            if show_plots == True:
                plt.show()
                st.pyplot()
            # if show_plots==True:
                # plt.show()
                # st.pyplot()

        def k_means(X, k, show_type='all', show_plots=True):
            c_prev = random_centroid(X, k)
            cluster = assign_cluster(X, c_prev, k)
            diff = 10
            ini_centroid = c_prev

            
            while diff > 0.0001:
                cluster = assign_cluster(X, c_prev, k)
                if show_type == 'all' and show_plots:
                    show_clusters(X, cluster, c_prev, ini_centroid,
                                  False, False, show_plots=show_plots)
                    mark_centroid = False
                    show_ini_centroid = False
                c_new = compute_centroid(X, cluster, k)
                diff = difference(c_prev, c_new)
                c_prev = c_new
            if show_plots:
                # st.write("Initial Cluster Centers:")
                # st.write(ini_centroid)
                # st.write("Final Cluster Centers:")
                # st.write(c_prev)
                # st.write("Final Plot:")
                show_clusters(X, cluster, c_prev, ini_centroid,
                              mark_centroid=True, show_ini_centroid=True)
            return cluster, c_prev

        def validate(original_clus, my_clus, k):
            ori_grp = []
            my_grp = []
            for i in range(k):
                temp = []
                temp1 = []
                for j in range(len(my_clus)):
                    if my_clus[j] == i:
                        temp.append(j)
                    if original_clus[j] == i:
                        temp1.append(j)
                my_grp.append(temp)
                ori_grp.append(temp1)
            same_bool = True
            for f in range(len(ori_grp)):
                if my_grp[f] not in ori_grp:
                    st.write("Not Same")
                    same_bool = False
                    break
            if same_bool:
                st.write("Both the clusters are equal")
        k = st.number_input("Enter value for K", step=1, value=1)
        X, original_clus = make_blobs(
            n_samples=50, centers=3, n_features=2, random_state=len(attribute1))
        datat = []
        arr1 = []
        arr2 = []
        for i in range(len(df)):
            arr1.append(df.loc[i, attribute1])
        for i in range(len(df)):
            arr2.append(df.loc[i, attribute2])
        for i in range(len(arr1)):
            tmp = []
            tmp.append(arr1[i])
            tmp.append(arr2[i])
            datat.append(tmp)
        cluster, centroid = k_means(datat, k, show_type='ini_fin')



    if rad == "Assignment 7":
        def app(dataset):

            st.header("Assignment 7")

            url = 'https://raw.githubusercontent.com/Udayraj2806/dataset/main/house-votes-84.data.csv'
            df = pd.read_csv(url)

            # st.write(df[:5])
            d = pd.DataFrame(df)
            data = d
            d.head()

            df_rows = d.to_numpy().tolist()

            cols = []
            for i in data.columns:
                cols.append(i)
            st.write(cols)
            col_len = len(cols)
            st.write("At Max Rules to be Generated: ",
                    ((3*col_len)-(2*(col_len+1)))+1)
            st.write("Attributes:", len(cols))
            # st.write(cols)
            newDataSet = []
            # st.write(len(df_rows))
            i, cnt = 0, 0
            for row in df_rows:
                i += 1
                if '?' in row:
                    continue
                else:
                    lst = []
                    cnt += 1
                    for k in range(1, len(row)):
                        if row[k] == 'y':
                            lst.append(cols[k])
                    newDataSet.append(lst)
            # st.write(newDataSet)

            # st.write(row)
            # st.write("--------------")
            # st.write(cnt)
            # st.write(newDataSet)
            # newDataSet.drop()

            data = []

            for i in range(len(newDataSet)):
                # data[i] = newDataSet[i]
                data.append([i, newDataSet[i]])

            # st.write(data)

            # extract distinct items

            init = []
            for i in data:
                for q in i[1]:
                    if (q not in init):
                        init.append(q)
            init = sorted(init)

            st.write("Init:", len(init))

            # st.write(init)

            sp = 0.4
            s = int(sp*len(init))
            s

            c = Counter()
            for i in init:
                for d in data:
                    if (i in d[1]):
                        c[i] += 1
            # st.write("C1:")
            for i in c:
                pass
                # st.write(str([i])+": "+str(c[i]))
            # st.write()
            l = Counter()
            for i in c:
                if (c[i] >= s):
                    l[frozenset([i])] += c[i]
            # st.write("L1:")
            for i in l:
                pass
                # st.write(str(list(i))+": "+str(l[i]))
            # st.write()
            pl = l
            pos = 1
            for count in range(2, 1000):
                nc = set()
                temp = list(l)
                for i in range(0, len(temp)):
                    for j in range(i+1, len(temp)):
                        t = temp[i].union(temp[j])
                        if (len(t) == count):
                            nc.add(temp[i].union(temp[j]))
                nc = list(nc)
                c = Counter()
                for i in nc:
                    c[i] = 0
                    for q in data:
                        temp = set(q[1])
                        if (i.issubset(temp)):
                            c[i] += 1
                # st.write("C"+str(count)+":")
                for i in c:
                    pass
                    # st.write(str(list(i))+": "+str(c[i]))
                # st.write()
                l = Counter()
                for i in c:
                    if (c[i] >= s):
                        l[i] += c[i]
                # st.write("L"+str(count)+":")
                for i in l:
                    pass
                    # st.write(str(list(i))+": "+str(l[i]))
                # st.write()
                if (len(l) == 0):
                    break
                pl = l
                pos = count
            st.write("Result: ")
            st.write("L"+str(pos)+":")

            for i in pl:
                st.write(str(list(i))+": "+str(pl[i]))

            st.subheader("Rules Generation")
            for l in pl:
                st.write(l)
                break
            from itertools import combinations
            for l in pl:
                cnt = 0
                c = [frozenset(q) for q in combinations(l, len(l)-1)]
                mmax = 0
                for a in c:
                    b = l-a
                    ab = l
                    sab = 0
                    sa = 0
                    sb = 0
                    for q in data:
                        temp = set(q[1])
                        if (a.issubset(temp)):
                            sa += 1
                        if (b.issubset(temp)):
                            sb += 1
                        if (ab.issubset(temp)):
                            sab += 1
                    temp = sab/sa*100
                    if (temp > mmax):
                        mmax = temp
                    temp = sab/sb*100
                    if (temp > mmax):
                        mmax = temp
                    cnt += 1
                    st.write(str(cnt) + str(list(a))+" -> " +
                            str(list(b))+" = "+str(sab/sa*100)+"%")
                    cnt += 1
                    st.write(str(cnt) + str(list(b))+" -> " +
                            str(list(a))+" = "+str(sab/sb*100)+"%")
                mmax = st.number_input('Select value of alpha', step=5, min_value=5)
                mmax = int(mmax)
                curr = 1
                st.write("choosing:", end=' ')

                for a in c:
                    b = l-a
                    ab = l
                    sab = 0
                    sa = 0
                    sb = 0
                    for q in data:
                        temp = set(q[1])
                        if (a.issubset(temp)):
                            sa += 1
                        if (b.issubset(temp)):
                            sb += 1
                        if (ab.issubset(temp)):
                            sab += 1
                    temp = sab/sa*100
                    if (temp >= mmax):
                        st.write(curr, end=' ')
                    curr += 1
                    temp = sab/sb*100
                    if (temp >= mmax):
                        st.write(curr, end=' ')
                    curr += 1
                    # break
                st.write()
                st.write()
                break

        app("")

        ####k-medoids
        

