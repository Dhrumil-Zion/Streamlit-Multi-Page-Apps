from sys import builtin_module_names
from pandas.core.algorithms import mode
import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('Agg')


from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def eda():
	st.subheader("Exploratory Data Analysis")
	data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
	flag = 0
	if data is not None:
		df = pd.read_csv(data)
		st.dataframe(df.head())

		if st.checkbox("Show Shape"):
			st.text("(Rows , Columns) of Dataset")
			st.write(df.shape)

		if st.checkbox("Show Columns"):
			flag = 1
			st.text("List of all Columns in Dataset")
			all_columns = df.columns.to_list()
			st.write(all_columns)

		if st.checkbox("Summary"):
			st.text("It shows many pre-calculate prameters to have and overview of dataset in a tabular form")
			st.write(df.describe())

		if st.checkbox("Show Selected Columns"):
			if flag:
				selected_columns = st.multiselect("Select Columns",all_columns)
				new_df = df[selected_columns]
				st.dataframe(new_df)
			else:
				st.warning("Please Check Show Columns First ")	

		if st.checkbox("Show Value Counts"):
			st.write(df.iloc[:,-1].value_counts())

		if st.checkbox("Correlation Plot(Matplotlib)"):
			plt.matshow(df.corr())
			st.set_option('deprecation.showPyplotGlobalUse', False)
			st.pyplot()

		if st.checkbox("Correlation Plot(Seaborn)"):
			st.write(sns.heatmap(df.corr(),annot=True))
			st.pyplot()

		if st.checkbox("Pie Plot"):
			all_columns = df.columns.to_list()
			column_to_plot = st.selectbox("Select 1 Column",all_columns)
			pie_plot = df[column_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
			st.write(pie_plot)
			st.pyplot()

def ploting():
	st.subheader("Data Visualization")
	data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
	if data is not None:
		df = pd.read_csv(data)
		st.dataframe(df.head())


		if st.checkbox("Show Value Counts"):
			st.write(df.iloc[:,-1].value_counts().plot(kind='bar'))
			st.set_option('deprecation.showPyplotGlobalUse', False)
			st.pyplot()
	

		all_columns_names = df.columns.tolist()
		type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
		selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)

		if st.button("Generate Plot"):
			st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))

			if type_of_plot == 'area':
				cust_data = df[selected_columns_names]
				st.area_chart(cust_data)

			elif type_of_plot == 'bar':
				cust_data = df[selected_columns_names]
				st.bar_chart(cust_data)

			elif type_of_plot == 'line':
				cust_data = df[selected_columns_names]
				st.line_chart(cust_data)


			elif type_of_plot:
				cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
				st.write(cust_plot)
				st.pyplot()


def ml():
	st.subheader("Summary of Different ML Models")
	data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
	if data is not None:
		df = pd.read_csv(data)
		st.dataframe(df.head())

		X = df.iloc[:,0:-1]
		Y = df.iloc[:,-1]
		seed = 7

		models = [('LR', LogisticRegression())]
		models.append(('LDA', LinearDiscriminantAnalysis()))
		models.append(('KNN', KNeighborsClassifier()))
		models.append(('CART', DecisionTreeClassifier()))
		models.append(('NB', GaussianNB()))
		models.append(('SVM', SVC()))

		model_names = []
		model_mean = []
		model_std = []
		all_models = []
		scoring = 'accuracy'
		for name, model in models:
			kfold = model_selection.KFold(n_splits=10, random_state=seed,shuffle=True)
			cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
			model_names.append(name)
			model_mean.append(cv_results.mean())
			model_std.append(cv_results.std())
			accuracy_results = {"model name":name,"model_accuracy":cv_results.mean(),"standard deviation":cv_results.std()}
			all_models.append(accuracy_results)


		if st.checkbox("Metrics As Table"):
			st.dataframe(pd.DataFrame(zip(model_names,model_mean,model_std),columns=["Algo","Mean of Accuracy","Std"]))

		if st.checkbox("Metrics As JSON"):
			st.json(all_models)


def app():

	st.header("Data Visualiaztion & Semi-Auto Machine Learning App")
	
	activities = ["Eploratory Data Analysis","Graph Analytics","Machine Learning"]	
	choice = st.sidebar.selectbox("Select Activities",activities)
	
	if choice == 'Eploratory Data Analysis':
		eda()	
	elif choice == 'Graph Analytics':
		ploting()
		
	elif choice == 'Machine Learning':
		ml()		
