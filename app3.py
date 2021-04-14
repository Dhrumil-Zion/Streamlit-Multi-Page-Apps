import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import numpy as np
import plotly.express as px


def get_dataset(dataset_name):
    if dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name == 'Breast Cancer':
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()    
    X = data.data
    y = data.target
    return X, y    

def add_parameter(class_if):
    params = {}
    if class_if == 'KNN':
        K = st.sidebar.slider("K",1,11)
        params["K"] = K
    elif class_if == 'SVM':
        C = st.sidebar.slider("C",0.01,10.0)
        params["C"] = C    
    else:
        depth = st.sidebar.slider("depth",2,50)
        number_of_trees = st.sidebar.slider("number_of_trees",1,100) 
        params["depth"] = depth
        params["number_of_trees"]  = number_of_trees
    return params    


def classifier(class_if,params):
    if class_if == 'KNN':
        return KNeighborsClassifier(n_neighbors=params["K"])
    elif class_if == 'SVM':
        return SVC(C=params["C"])
    else:
        return RandomForestClassifier(n_estimators =
        params["number_of_trees"],max_depth=params["depth"],random_state=1234)    

def app():
	st.title("Different Machine Learning Algorithms")

	dataset_name = st.sidebar.selectbox("Select the Dataset",("Iris","Breast Cancer","Wine Dataset"))
	classifier_name = st.sidebar.selectbox("Select the Classifier",("KNN","SVM","Random Forest"))
	
	X , y = get_dataset(dataset_name)
	st.write("Shape of datasets",X.shape)
	st.write("Number of classes",len(np.unique(y)))

	params = add_parameter(classifier_name)

	clf = classifier(classifier_name,params)
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1234)
	clf.fit(X_train,y_train)
	y_pred = clf.predict(X_test)
	acc = accuracy_score(y_test,y_pred)

	st.write(f"Classifier {classifier_name}")
	st.write(f"Accuracy is {acc}")

	pca = PCA(2)
	X_projected = pca.fit_transform(X)

	x1 = X_projected[:,0]
	x2 = X_projected[:,1]

	fig = px.scatter(x1,x2)

	st.plotly_chart(fig)

