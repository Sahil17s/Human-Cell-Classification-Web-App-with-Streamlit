import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

def main():
    st.sidebar.title("Human Cells Classification Web App")
    html_temp = """<div style="background-color:rgb(60, 79, 136);"><p style="color:white;font-size:50px;padding:10px">Human Cells Classification Web App</p></div>"""
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Classify human cells as Benign or Malignant")
    st.sidebar.markdown("Classify human cells as Benign or Malignant!")

    @st.cache(persist=True)
    def load_data():
        data=pd.read_csv("cell_samples.csv")
        data = data[pd.to_numeric(data['BareNuc'], errors='coerce').notnull()]
        data['BareNuc'] = data['BareNuc'].astype('int')
        return data 
    
    df=load_data()
    
    if st.sidebar.checkbox('Show raw data',False):
        st.subheader("Cell Samples Dataset for classification ")
        st.write(df)


    if st.sidebar.checkbox("Show Distribution plot of classes",False):
        st.subheader("Distribution of the classes of cell size")
        plt.figure(figsize=(6,3))
        ax = df[df['Class'] == 4].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant',figsize=(10,6))
        df[df['Class'] == 2].plot(kind='scatter', x='Clump', y='UnifSize', color='red', label='benign', ax=ax)
        plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
        plt.tight_layout()
        st.pyplot()
    
    @st.cache(persist=True)
    def split_data(df):
        feature_df = df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
        X = np.asarray(feature_df)
        df['Class'] = df['Class'].astype('int')
        y = np.asarray(df['Class'])
        X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.3,random_state=4)
        return X_train, X_test, y_train, y_test 
    
    X_train,X_test,y_train,y_test=split_data(df)
    class_names=['Malignant','Benign']
    st.sidebar.subheader("Choose Classifier")
    classifier= st.sidebar.selectbox("Classifier",('SVM','Logistic Regression', 'Random Forest'))

    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader('Confusion Matrix')
            plot_confusion_matrix(model,X_test,y_test,display_labels= class_names)
            st.pyplot()
        
        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model,X_test,y_test)
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model,X_test,y_test)
            st.pyplot()
    

    
    if classifier=='SVM':
        st.sidebar.subheader('Model Hyperparamteres')
        C=st.sidebar.number_input("C (Regularization parameter)",0.01,10.0,step=0.01,key='C')
        kernel= st.sidebar.radio("Kernel",("rbf","linear","sigmoid"),key='kernel')
        gamma=st.sidebar.radio("Gamma",("scale","auto"),key='gamma')

        metrics =st.sidebar.multiselect("Select metrics",("Confusion Matrix",'ROC Curve',"Precision-Recall Curve"))
        
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            model=SVC(C=C,kernel=kernel,gamma=gamma)
            model.fit(X_train,y_train)
            accuracy= model.score(X_test,y_test)
            y_pred=model.predict(X_test)
            st.write('Accuracy: ', accuracy.round(2))
            st.write("Precision: ", precision_score(y_test,y_pred,pos_label=2,labels=class_names).round(2))
            st.write("Recall: ",recall_score(y_test,y_pred,pos_label=2,labels=class_names).round(2))
            plot_metrics(metrics)
        
    if classifier=='Logistic Regression':
        st.sidebar.subheader('Model Hyperparamteres')
        C=st.sidebar.number_input("C (Regularization parameter)",0.01,10.0,step=0.01,key="C_LR")
        max_iter= st.sidebar.slider("Maximum no. of iterations",100,500,key='max_iter')


        metrics =st.sidebar.multiselect("Select metrics",("Confusion Matrix",'ROC Curve',"Precision-Recall Curve"))
        
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            model=LogisticRegression(C=C,max_iter=max_iter)
            model.fit(X_train,y_train)
            accuracy= model.score(X_test,y_test)
            y_pred=model.predict(X_test)
            st.write('Accuracy: ', accuracy.round(2))
            st.write("Precision: ", precision_score(y_test,y_pred,pos_label=2,labels=class_names).round(2))
            st.write("Recall: ",recall_score(y_test,y_pred,pos_label=2,labels=class_names).round(2))
            plot_metrics(metrics)
    
    if classifier=='Random Forest':
        st.sidebar.subheader('Model Hyperparamteres')
        n_estimators=st.sidebar.number_input("No. of trees in forest",100,500,step=10,key='n_estimators')
        max_depth=st.sidebar.number_input("Maximum depth of tree",1,20,step=1,key='max_depth')
        bootstrap=st.sidebar.radio("Bootstrap samples while building trees",("True","False"),key='bootstrap')


        metrics =st.sidebar.multiselect("Select metrics",("Confusion Matrix",'ROC Curve',"Precision-Recall Curve"))
        
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            model=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,bootstrap=bootstrap,n_jobs=-1)
            model.fit(X_train,y_train)
            accuracy= model.score(X_test,y_test)
            y_pred=model.predict(X_test)
            st.write('Accuracy: ', accuracy.round(2))
            st.write("Precision: ", precision_score(y_test,y_pred,pos_label=2,labels=class_names).round(2))
            st.write("Recall: ",recall_score(y_test,y_pred,pos_label=2,labels=class_names).round(2))
            plot_metrics(metrics)      
    



if __name__ == '__main__':
    main()
