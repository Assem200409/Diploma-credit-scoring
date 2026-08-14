import streamlit as st 
import pandas as pd    
import numpy as np     
from pickle import load, loads 
#
# from assem.class_Preparing import Preparing
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout 
#st.title(r"$\Huge🇰🇿$")
st.title("Credit Scoring Alfa Bank(Russia) 2022")

my_data = st.sidebar.selectbox("Choose your dataset",
                               ["UCI_Credit_Card.csv"]
                               )
#st.sidebar.selectbox('Shold', min_value=0, max_value=1, step=0.01, format=None)
shold = st.sidebar.selectbox('shold',[0.5,0.55,0.6,0.65,0.70,0.75,0.8, 0.85])
#df = pd.read_csv(my_data)
df_min_max = pd.read_csv('1_df_min_max.csv')
df_min = df_min_max['min'].values
df_max = df_min_max['max'].values


#limit_ball = st.number_input('LIMIT_BALL', min_value=0, max_value=1000000, step=10000, format=None)        

with st.expander("Payment status"): # свертывающийся блок
        pd1 = st.selectbox("Month 1", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
        pd2 = st.selectbox("Month 2", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
        pd3 = st.selectbox("Month 3", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
        pd4 = st.selectbox("Month 4", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
        pd5 = st.selectbox("Month 5", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
        pd6 = st.selectbox("Month 6", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
        pd7 = st.selectbox("Month 7", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
        pd8 = st.selectbox("Month 8", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
        pd9 = st.selectbox("Month 9", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
        pd10 = st.selectbox("Month 10", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
        pd11 = st.selectbox("Month 11", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
        pd12 = st.selectbox("Month 12", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
        pd13 = st.selectbox("Month 13", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
        pd14 = st.selectbox("Month 14", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
        pd15 = st.selectbox("Month 15", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
        pd16 = st.selectbox("Month 16", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
        pd17 = st.selectbox("Month 17", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
        pd18 = st.selectbox("Month 18", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
        pd19 = st.selectbox("Month 19", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)

st.title("Predictions")

exp = [pd1, pd2, pd3, pd4, pd5, pd6, 
       pd7, pd8, pd9, pd10, pd11, pd12, 
       pd13, pd14, pd15, pd16, pd17, pd18, pd19]

# Преобразование данных клиента в соответствии с данными датасета
for i, v in enumerate(exp):
     if v<df_min[i]:
          exp[i] = df_min[i] + 0.01
     else:
          exp[i] = exp[i] + 0.01       
     if v>df_max[i]:
          exp[i] = df_max[i] - 0.01

if st.button("My predictions", disabled = False ): # кнопка предсказаний, кнопка активна

    with open("./dtree.pkl", "rb") as f: # rb - режим чтения, ./model.pkl - файл куда сохраняется модель
        dtree = load(f)                  # загруженная модель dtree
        pred_proba = dtree.predict_proba([exp])[0]
        p_no_default = pred_proba[0]
        p_default = pred_proba[1]

        if p_no_default > shold:
            pred_tree = 0
            st.write("Decision Tree")
            st.success("0 -- no default")
            st.success(round(p_no_default, 4))

        else:
            pred_tree = 1
            st.write("Decision Tree")
            st.error("1 -- default")
            st.error(round(p_default, 4)) 
    
    with open("./rfc.pkl", "rb") as f: # rb - режим чтения, ./model.pkl - файл куда сохраняется модель
        rfc = load(f)                  # загруженная модель rfc

        pred_proba = rfc.predict_proba([exp])[0]
        p_no_default = pred_proba[0]
        p_default = pred_proba[1]
  
        if p_no_default > shold:
            pred_rfc = 0
            st.write("Random Forest")
            st.success("0 -- no default")
            st.success(round(p_no_default, 4))
        else:
            pred_rfc = 1
            st.write("Random Forest")
            st.error("1 -- default")
            st.error(round(p_default, 4)) 

    with open("./boost_gr.pkl", "rb") as f: # rb - режим чтения, ./model.pkl - файл куда сохраняется модель
        boost_gr = load(f)                  # загруженная модель boost_gr
        pred_proba = boost_gr.predict_proba([exp])[0]
        p_no_default = pred_proba[0]
        p_default = pred_proba[1]
        
        if p_no_default>shold:
            pred_boost_gr = 0
            st.write("Gradient Boosting")
            st.success("0 -- no default")
            st.success(round(p_no_default, 4))

        else:  
            pred_boost_gr = 1  
            st.write("Gradient Boosting")
            st.error("1 -- default")
            st.error(round(p_default, 4))  

    model = keras.models.load_model("my_model_3mln.h5", compile=False) 
   
    exp_nn = np.array(exp).reshape(1, 19)
    pred_proba_nn = model.predict([exp_nn])[0][0]

    p = pred_proba_nn
    p1, p2 = 1-p, p

    if  p1 > shold:
        pred_nn = 0
        st.write("Neural Network")
        st.success("0 -- no default")
        st.success(round(1-pred_proba_nn, 4))

    else:  
        pred_nn = 1  
        st.write("Neural Network")
        st.error("1 -- default")
        st.error(round(pred_proba_nn, 4))

    
    
