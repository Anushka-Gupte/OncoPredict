import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

model = pickle.load(open('onco-predict.pkl','rb'))

st.header('Breast Cancer Risk Prediction System')

st.subheader('Global Feature Importance')

from sklearn.model_selection import train_test_split
dataset = pd.read_csv('./data/breast_cancer.csv')
X = dataset.iloc[:,1:-1]
y = dataset.iloc[:,-1].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
explainer = shap.Explainer(model,X_train)
shap_values = explainer(X_train)
fig,ax = plt.subplots()
shap.plots.bar(shap_values,max_display=10,show=False)
st.pyplot(fig)
st.write(""" 
        This plot shows which features strongly influence the model's prediction overall. 
         Larger mean SHAP values indicate features like Bare Nuclei and Clump Thickness are the 
         most important drivers in distinguishing benign from malignant tumors.
        """)

st.subheader('Patient Input')

ct = st.number_input('Clump Thickness',step=1)
cs = st.number_input('Uniformity of Cell Size',step=1)
cshape = st.number_input('Uniformity of Cell Shape',step=1)
ma = st.number_input('Marginal Adhesion',step=1)
sc = st.number_input('Single Epithelial Cell Size',step=1)
bn = st.number_input('Bare Nuclei',step=1)
bc = st.number_input('Bland Chromatin',step=1)
nn = st.number_input('Normal Nucleoli',step=1)
m = st.number_input('Mitoses',step=1)

st.subheader('Prediction Result')

if st.button('Predict'):
    res = model.predict([[ct,cs,cshape,ma,sc,bn,bc,nn,m]])
    st.metric('Class ',int(res))
    if(int(res) == 2):
        st.subheader('Benign')
    else:
        st.subheader('Malignant')

    feature_names = ['Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape',
                     'Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei',
                     'Bland Chromatin','Normal Nucleoli','Mitoses']
    input_df = pd.DataFrame([[ct,cs,cshape,ma,sc,bn,bc,nn,m]],columns=feature_names)
    st.subheader('Explanation')
    shap_values_single =  explainer(input_df)
    fig,ax = plt.subplots()
    shap.plots.waterfall(shap_values_single[0],show=False)
    st.pyplot(fig)
    st.write(""" 
            This plot illustrates how each tumor feature influenced the model's prediction for this patient.
             Features pushing the prediction towards malignancy are shown in red, while those reducing 
             malignancy risk are shown in blue, providing transperant, patient-specific interpretability.
            """)
