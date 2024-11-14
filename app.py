import pickle
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
#st.set_option('deprecation.showPyplotGlobalUse', False)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.pyplot")

model=pickle.load(open("rf.pkl", "rb"))

def main():
    st.sidebar.title('Diabetes Risk Prediction')
    age=st.sidebar.selectbox("Age",['10-20','20-30','30-40','40-50','50-60','60-70','70-80','80-90','90-100'])
    gender=st.sidebar.selectbox("Gender",['Male','Female'])
    polyuria=st.sidebar.selectbox("Polyuria",['Yes','No'])
    polydipsia=st.sidebar.selectbox("Polydipsia",['Yes','No'])
    s_w_l=st.sidebar.selectbox("Suddent Weight Loss",['Yes','No'])
    weakness=st.sidebar.selectbox("Weakness",['Yes','No'])
    polyphagia=st.sidebar.selectbox("Polyphagia",['Yes','No'])
    g_t=st.sidebar.selectbox("Genital Thrush",['Yes','No'])
    v_b=st.sidebar.selectbox("Visual Blurring",['Yes','No'])
    itching=st.sidebar.selectbox("Itching",['Yes','No'])
    irritability=st.sidebar.selectbox("Irritability",['Yes','No'])
    d_h=st.sidebar.selectbox("Delayed Healing",['Yes','No'])
    p_p=st.sidebar.selectbox("Partial Paresis",['Yes','No'])
    m_s=st.sidebar.selectbox("Muscle Stiffness",['Yes','No'])
    alopecia=st.sidebar.selectbox("Alopecia",['Yes','No'])
    obesity=st.sidebar.selectbox("Obesity",['Yes','No'])
    
    df = pd.DataFrame(data=[[gender, polyuria, polydipsia, s_w_l, weakness, polyphagia, g_t, v_b, itching, irritability, d_h, p_p, m_s, alopecia, obesity, age]], 
                      columns=['Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring',
       'Itching', 'Irritability', 'delayed healing', 'partial paresis',
       'muscle stiffness', 'Alopecia', 'Obesity','age_group'])
    df['age_group'].replace(['10-20','20-30','30-40','40-50','50-60','60-70','70-80','80-90'], [0,1,2,3,4,5,6,7], inplace=True)
    df['Gender'].replace(['Female', 'Male'], [0, 1], inplace=True)
    df['Polyuria'].replace(['No', 'Yes'], [0, 1], inplace=True)
    df['Polydipsia'].replace(['No', 'Yes'], [0, 1], inplace=True)
    df['sudden weight loss'].replace(['No', 'Yes'], [0, 1], inplace=True)
    df['weakness'].replace(['No', 'Yes'], [0, 1], inplace=True)
    df['Polyphagia'].replace(['No', 'Yes'], [0, 1], inplace=True)
    df['Genital thrush'].replace(['No', 'Yes'], [0, 1], inplace=True)
    df['visual blurring'].replace(['No', 'Yes'], [0, 1], inplace=True)
    df['Itching'].replace(['No', 'Yes'], [0, 1], inplace=True)
    df['Irritability'].replace(['No', 'Yes'], [0, 1], inplace=True)
    df['delayed healing'].replace(['No', 'Yes'], [0, 1], inplace=True)
    df['partial paresis'].replace(['No', 'Yes'], [0, 1], inplace=True)
    df['muscle stiffness'].replace(['No', 'Yes'], [0, 1], inplace=True)
    df['Alopecia'].replace(['No', 'Yes'], [0, 1], inplace=True)
    df['Obesity'].replace(['No', 'Yes'], [0, 1], inplace=True)

    if st.sidebar.button("Predict"):
        prediction = model.predict(df)
        if prediction == 0:
            st.sidebar.error("Negative")
        else:
            st.sidebar.success("Positive")
            
    html_temp = """ <div style="padding:1.5px">
                    <h1 style="color:black;text-align:center;">Early Stage Diabetes Risk Prediction</h1></div><br>"""
    st.markdown(html_temp,unsafe_allow_html=True)
    
    
    data = load_data()
    if st.checkbox("Dataset Analyze"):
        select1 = st.selectbox("Please select a section:", ["", "Head", "Describe"])
     
        if select1 == "Head":
            st.table(data.head())
        elif select1 == "Describe":
            select2 = st.selectbox("Please select value type:", ["", "Numerical", "Categorical"])
            if select2 == "Numerical":
                st.table(data.describe())
            elif select2 == "Categorical":
                st.table(data[['Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring',
       'Itching', 'Irritability', 'delayed healing', 'partial paresis',
       'muscle stiffness', 'Alopecia', 'Obesity','class']].describe())
    if st.checkbox("Visualization"):
            left_names=data["class"].value_counts().index
            left_val=data["class"].value_counts().values
            plt.title("Visualization of class")
            plt.pie(left_val, labels=left_names, autopct="''%1.2f%%'")
            plt.title("Visualization of class")
            st.pyplot()
            
            def age(i):
                for x in range(10,100,10):
                    if i<x:
                        m=f'{x-10}-{x}'
                        return m
            visual=data.copy()
            visual['age_group']=data['Age'].apply(lambda x:age(x))
            plt.title("Visualization of Age")
            fig,ax=plt.subplots(nrows=1, ncols=2, figsize=(10,4))
            sns.countplot(x='age_group', data=visual, ax=ax[0])
            ax[0].set_title("Age Wise Count")
            sns.countplot(x='age_group',hue='class',data=visual,ax=ax[1])
            ax[1].set_title("Age_Group Vs Class")
            st.pyplot()
            
            plt.title("Visualization of Gender")
            fig,ax=plt.subplots(nrows=1, ncols=2, figsize=(10,4))
            sns.countplot(x='Gender', data=data, ax=ax[0])
            ax[0].set_title("Gender Wise Count")
            sns.countplot(x='Gender',hue='class',data=data,ax=ax[1])
            ax[1].set_title("Gender Vs Class")
            st.pyplot()
            
            plt.title("Visualization of Polyuria")
            fig,ax=plt.subplots(nrows=1, ncols=2, figsize=(10,4))
            sns.countplot(x='Polyuria', data=data, ax=ax[0])
            ax[0].set_title("Polyuria Symptom Count")
            sns.countplot(x='Polyuria',hue='class',data=data,ax=ax[1])
            ax[1].set_title("Polyuria Vs Class")
            st.pyplot()
            
            plt.title("Visualization of Polydipsia")
            fig,ax=plt.subplots(nrows=1, ncols=2, figsize=(10,4))
            sns.countplot(x='Polydipsia', data=data, ax=ax[0])
            ax[0].set_title("Polydipsia Symptom Count")
            sns.countplot(x='Polydipsia',hue='class',data=data,ax=ax[1])
            ax[1].set_title("Polydipsia Vs Class")
            st.pyplot()
            
            plt.title("Visualization of Sudden Weight Loss")
            fig,ax=plt.subplots(nrows=1, ncols=2, figsize=(10,4))
            sns.countplot(x='sudden weight loss', data=data, ax=ax[0])
            ax[0].set_title("sudden weight loss Symptom Count")
            sns.countplot(x='sudden weight loss',hue='class',data=data,ax=ax[1])
            ax[1].set_title("sudden weight loss Vs Class")
            st.pyplot()
            
            plt.title("Visualization of Weakness")
            fig,ax=plt.subplots(nrows=1, ncols=2, figsize=(10,4))
            sns.countplot(x='weakness', data=data, ax=ax[0])
            ax[0].set_title("Weakness Symptom Count")
            sns.countplot(x='weakness',hue='class',data=data,ax=ax[1])
            ax[1].set_title("Weakness Vs Class")
            st.pyplot()
            
            plt.title("Visualization of Polyphagia")
            fig,ax=plt.subplots(nrows=1, ncols=2, figsize=(10,4))
            sns.countplot(x='Polyphagia', data=data, ax=ax[0])
            ax[0].set_title("Polyphagia Symptom Count")
            sns.countplot(x='Polyphagia',hue='class',data=data,ax=ax[1])
            ax[1].set_title("Polyphagia Vs Class")
            st.pyplot()
            
            plt.title("Visualization of Genital Thrush")
            fig,ax=plt.subplots(nrows=1, ncols=2, figsize=(10,4))
            sns.countplot(x='Genital thrush', data=data, ax=ax[0])
            ax[0].set_title("Genital thrush Symptom Count")
            sns.countplot(x='Genital thrush',hue='class',data=data,ax=ax[1])
            ax[1].set_title("Genital thrush Vs Class")
            st.pyplot()
            
            plt.title("Visualization of Visual Blurring")
            fig,ax=plt.subplots(nrows=1, ncols=2, figsize=(10,4))
            sns.countplot(x='visual blurring', data=data, ax=ax[0])
            ax[0].set_title("visual blurring Symptom Count")
            sns.countplot(x='visual blurring',hue='class',data=data,ax=ax[1])
            ax[1].set_title("visual blurring Vs Class")
            st.pyplot()
            
            plt.title("Visualization of Itching")
            fig,ax=plt.subplots(nrows=1, ncols=2, figsize=(10,4))
            sns.countplot(x='Itching', data=data, ax=ax[0])
            ax[0].set_title("Itching Symptom Count")
            sns.countplot(x='Itching',hue='class',data=data,ax=ax[1])
            ax[1].set_title("Itching Vs Class")
            st.pyplot()
            
            plt.title("Visualization of Irritability")
            fig,ax=plt.subplots(nrows=1, ncols=2, figsize=(10,4))
            sns.countplot(x='Irritability', data=data, ax=ax[0])
            ax[0].set_title("Irritability Symptom Count")
            sns.countplot(x='Irritability',hue='class',data=data,ax=ax[1])
            ax[1].set_title("Irritability Vs Class")
            st.pyplot()
            
            plt.title("Visualization of Delayed Healing")
            fig,ax=plt.subplots(nrows=1, ncols=2, figsize=(10,4))
            sns.countplot(x='delayed healing', data=data, ax=ax[0])
            ax[0].set_title("delayed healing Symptom Count")
            sns.countplot(x='delayed healing',hue='class',data=data,ax=ax[1])
            ax[1].set_title("delayed healing Vs Class")
            st.pyplot()
            
            plt.title("Visualization of Partial Paresis")
            fig,ax=plt.subplots(nrows=1, ncols=2, figsize=(10,4))
            sns.countplot(x='partial paresis', data=data, ax=ax[0])
            ax[0].set_title("partial paresis Symptom Count")
            sns.countplot(x='partial paresis',hue='class',data=data,ax=ax[1])
            ax[1].set_title("partial paresis Vs Class")
            st.pyplot()
            
            plt.title("Visualization of Muscle Stiffness")
            fig,ax=plt.subplots(nrows=1, ncols=2, figsize=(10,4))
            sns.countplot(x='muscle stiffness', data=data, ax=ax[0])
            ax[0].set_title("muscle stiffness Symptom Count")
            sns.countplot(x='muscle stiffness',hue='class',data=data,ax=ax[1])
            ax[1].set_title("muscle stiffness Vs Class")
            st.pyplot()
            
            plt.title("Visualization of Alopecia")
            fig,ax=plt.subplots(nrows=1, ncols=2, figsize=(10,4))
            sns.countplot(x='Alopecia', data=data, ax=ax[0])
            ax[0].set_title("Alopecia Symptom Count")
            sns.countplot(x='Alopecia',hue='class',data=data,ax=ax[1])
            ax[1].set_title("Alopecia Vs Class")
            st.pyplot()
            
            plt.title("Visualization of Obesity")
            fig,ax=plt.subplots(nrows=1, ncols=2, figsize=(10,4))
            sns.countplot(x='Obesity', data=data, ax=ax[0])
            ax[0].set_title("Obesity Symptom Count")
            sns.countplot(x='Obesity',hue='class',data=data,ax=ax[1])
            ax[1].set_title("Obesity Vs Class")
            st.pyplot()

def load_data():
    df = pd.read_csv("diabetes_data.csv")
    return df 

if __name__ == "__main__":
    main()
