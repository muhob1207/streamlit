import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, accuracy_score

st.set_page_config(page_title='Machine Learning Application')
st.header('Welcome to ML Application')
image1=Image.open('ML.png')

st.image(image1,use_column_width=True)

st.write("<div style='text-align:center'>Click on the homepage on the sidebar to the left to familiarize yourself with the data</div>", unsafe_allow_html=True)

st.sidebar.title("Pages")
page = st.sidebar.selectbox(
    'Select one of the pages',
    ('Homepage', 'Data Preparation', 'Modeling'))

st.write('\n')
st.write('\n')


if page == 'Homepage':
    data_choice = st.selectbox(
    'Select the data that you want to analyze',
    ('Water potability', 'Loan prediction'))
    if data_choice == 'Water potability':
        df = pd.read_csv('water_potability.csv')
    else:
        df = pd.read_csv('loan_pred.csv')
    st.markdown('Data')
    st.dataframe(df)
    st.markdown('Dataframe length')
    st.write(len(df))
    st.markdown('Data description')
    st.dataframe(df.describe())
    st.markdown('Number of null values')
    st.dataframe(df.isnull().sum())
    st.markdown('Presence of duplicates')
    st.write(len(df)!=len(df.drop_duplicates()))
    st.markdown('Balance of values of the label')
    st.dataframe(pd.DataFrame(df.iloc[:,-1].value_counts()))


    #Selecting all numerical columns
    numerical_columns = df.select_dtypes(include=[float, int]).columns.tolist()
    filtered_numerical_columns = [col for col in numerical_columns if len(df[col].unique()) > 20]


    #Creating boxplots for all numerical columns
    st.markdown('Showing outliers for numerical columns')
    fig, axes = plt.subplots(nrows=len(filtered_numerical_columns), figsize=(10, 10))
    for i, column in enumerate(filtered_numerical_columns):
        sns.boxplot(x=df[column], ax=axes[i])
    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()

    st.markdown('Analyzing multicolinearity')
    plt.figure(figsize=(10,10))
    st.pyplot(sns.heatmap(df.corr(),annot=True).figure)
    plt.clf()
    st.markdown('\n')
    st.markdown("Please move to the page 'Data Preparation'")

elif page == 'Data Preparation':
    data_choice = st.selectbox(
    'Select the data that you want to analyze',
    ('Water potability', 'Loan prediction'))
    if data_choice == 'Water potability':
        df = pd.read_csv('water_potability.csv')
    else:
        df = pd.read_csv('loan_pred.csv')
    
    column1,column2=st.columns(2)

    with column1:
        st.header('Categorical columns')
        cat_fillna_options = st.radio(
        'Choose the method to fill null values in categorical columns',
        ('mode', 'ffill', 'bfill'))
        remove_duplicates = st.checkbox('Remove duplicates')

    with column2:
        st.header('Numerical columns')
        num_fillna_options = st.radio(
        'Choose the method to fill null values in numerical columns',
        ('mean', 'median', 'mode'))
        remove_outliers = st.checkbox('Remove outliers')

    numerical_columns = df.select_dtypes(include=[float, int]).columns.tolist()
    #Removing the categorical variables falsely interpreted as numerical
    filtered_numerical_columns = [col for col in numerical_columns if len(df[col].unique()) > 20]
    categorical_columns = [col for col in df.columns if col not in filtered_numerical_columns]

    #Removing the label from the categorical columns if it is there
    if list(df.columns)[-1] in categorical_columns:
        categorical_columns.remove(list(df.columns)[-1])
    
    #Else removing the label from the numerical columns list
    else:
        filtered_numerical_columns.remove(list(df.columns)[-1])

    if st.button('Prepare Data')==True:
        if len(categorical_columns)>0:
            if cat_fillna_options == 'mode':
                cat_imputer = SimpleImputer(strategy='most_frequent')
                df[categorical_columns] = cat_imputer.fit_transform(df[categorical_columns])
            elif cat_fillna_options == 'ffill':
                df[categorical_columns].fillna(method = 'ffill')
            else:
                df[categorical_columns].fillna(method = 'bfill')

        if len(filtered_numerical_columns) > 0:
            if num_fillna_options == 'mode':
                num_imputer = SimpleImputer(strategy='most_frequent')
                df[filtered_numerical_columns] = num_imputer.fit_transform(df[filtered_numerical_columns])
            elif num_fillna_options == 'median':
                num_imputer = SimpleImputer(strategy='median')
                df[filtered_numerical_columns] = num_imputer.fit_transform(df[filtered_numerical_columns])
            else:
                num_imputer = SimpleImputer(strategy='mean')
                df[filtered_numerical_columns] = num_imputer.fit_transform(df[filtered_numerical_columns])

            if remove_outliers == True:
                columns = filtered_numerical_columns
                def outlier_detection(cols):
                    Q1,Q3 = np.percentile(cols,[25,75])
                    IQR = Q3-Q1
                    upper_bound = Q3 + (1.5*IQR)
                    lower_bound = Q1 - (1.5*IQR)
                    return upper_bound,lower_bound
                for cols in columns:
                    upper_bound,lower_bound = outlier_detection(df[cols])
                    df[cols] = np.clip(df[cols],a_min=lower_bound,a_max=upper_bound)
        
        if remove_duplicates == True:
            df = df.drop_duplicates()
        
        #Removing the rows with remaining null values
        df = df.dropna(axis=0)

        st.markdown('Results of data preprocessing')
        st.markdown('Number of null values')
        st.dataframe(df.isnull().sum())
        st.markdown('Presence of duplicates')
        st.write(len(df)!=len(df.drop_duplicates()))
        st.markdown('Showing the prepared data')
        st.dataframe(df)
        st.markdown('Dataframe length')
        st.write(len(df))

        if remove_outliers == True and len(filtered_numerical_columns)>0: 
            fig, axes = plt.subplots(nrows=len(filtered_numerical_columns), figsize=(10, 10))
            for i, column in enumerate(filtered_numerical_columns):
                sns.boxplot(x=df[column], ax=axes[i])
            plt.tight_layout()
            st.pyplot(fig)
            plt.clf()
        if os.path.exists('prepared_data.csv') == True:
            os.remove('prepared_data.csv')
            df.to_csv('prepared_data.csv')
        else:
            df.to_csv('prepared_data.csv')
else:
    df = pd.read_csv('prepared_data.csv').iloc[:,1:]
    #splitting into features and labels
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    numerical_columns = df.select_dtypes(include=[float, int]).columns.tolist()
    #Removing the categorical variables falsely interpreted as numerical
    filtered_numerical_columns = [col for col in numerical_columns if len(df[col].unique()) > 20]
    categorical_columns = [col for col in df.columns if col not in filtered_numerical_columns]

    #Removing the label from the categorical columns if it is there
    if list(df.columns)[-1] in categorical_columns:
        categorical_columns.remove(list(df.columns)[-1])
    
    #Else removing the label from the numerical columns list
    else:
        filtered_numerical_columns.remove(list(df.columns)[-1])

    #Getting dummies for the label if its data type is object
    if str(df[list(df.columns)[-1]].dtype) == 'object':
        y = pd.get_dummies(y,drop_first=True) 
        y = y.iloc[:,0]

    
    tts = st.text_input('Enter the size of test data. For example 0.2')
    random_state = st.text_input('Enter the random state. For example 42')

    column1,column2=st.columns(2)

    with column1:
        st.header('Categorical columns')
        cat_encoding = st.radio(
        'Choose the method of encoding',
        ('OneHotEncoder', 'OrdinalEncoder'))

    with column2:
        st.header('Numerical columns')
        num_scaling = st.radio(
        'Choose the method of scaling',
        ('MinMaxScaler', 'StandardScaler', 'RobustScaler'))

    model_choice = st.radio(
    'Choose the model',
    ('SVM', 'Logistic Regression', 'Random Forest'))

    if st.button('Build model')==True:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(tts), random_state=int(random_state))


        if len(categorical_columns)>0:
            if cat_encoding == 'OneHotEncoder':
                encoder = OneHotEncoder(handle_unknown='ignore')
                X_train_encoded = encoder.fit_transform(X_train[categorical_columns])
                X_test_encoded = encoder.transform(X_test[categorical_columns])
                X_train_encoded = X_train_encoded.toarray()
                X_test_encoded = X_test_encoded.toarray()
                X_train = pd.concat([X_train.drop(categorical_columns, axis=1).reset_index().drop('index',axis=1), pd.DataFrame(X_train_encoded)], axis=1)
                X_test = pd.concat([X_test.drop(categorical_columns, axis=1).reset_index().drop('index',axis=1), pd.DataFrame(X_test_encoded)], axis=1)
            else:
                encoder = OrdinalEncoder()
                X_train[categorical_columns] = encoder.fit_transform(X_train[categorical_columns])
                X_test[categorical_columns] = encoder.transform(X_test[categorical_columns])
                
        if len(filtered_numerical_columns) > 0:
            if num_scaling == 'MinMaxScaler':
                scaler = MinMaxScaler()
                X_train[filtered_numerical_columns] = scaler.fit_transform(X_train[filtered_numerical_columns])
                X_test[filtered_numerical_columns] = scaler.transform(X_test[filtered_numerical_columns])
            elif num_scaling == 'StandardScaler':
                scaler = StandardScaler()
                X_train[filtered_numerical_columns] = scaler.fit_transform(X_train[filtered_numerical_columns])
                X_test[filtered_numerical_columns] = scaler.transform(X_test[filtered_numerical_columns])
            else:
                scaler = RobustScaler()
                X_train[filtered_numerical_columns] = scaler.fit_transform(X_train[filtered_numerical_columns])
                X_test[filtered_numerical_columns] = scaler.transform(X_test[filtered_numerical_columns])
        

        if model_choice == 'SVM':
            model = svm.SVC(kernel='poly',probability=True,random_state=int(random_state))
            model.fit(X_train, y_train)
        elif model_choice == 'Logistic Regression':
            model = LogisticRegression(max_iter = 100000,random_state=int(random_state))
            model.fit(X_train, y_train)
        else:
            model = RandomForestClassifier(n_estimators = 100,random_state=int(random_state))
            model.fit(X_train,y_train)

        y_pred = model.predict(X_test)

        accuracy = round(accuracy_score(y_test, y_pred),2)

        st.write('The accuracy score of the model is: ', accuracy)

        st.write('Confusion matrix')

        st.write(confusion_matrix(y_test, y_pred))

        st.write('ROC AUC SCORE')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        fig, ax = plt.subplots(figsize=(10,10))
        logit_roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
        fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
        plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        st.pyplot(fig)
        plt.clf()

