import streamlit as st 
import pandas as pd    
import numpy as np     
import os
import json
from pickle import load, loads 
from openai import OpenAI
from pathlib import Path 
import shap
# from assem.class_Preparing import Preparing
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout 

BASE_DIR = Path(__file__).resolve().parent

llm_client = OpenAI(base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
                timeout=30)

st.title("Credit Scoring Alfa Bank 2022")

my_data = st.sidebar.selectbox("Choose your dataset",
                               ["UCI_Credit_Card.csv"]
                               )
#st.sidebar.selectbox('Shold', min_value=0, max_value=1, step=0.01, format=None)
shold = st.sidebar.selectbox('shold',[0.5,0.55,0.6,0.65,0.70,0.75,0.8, 0.85])
#df = pd.read_csv(my_data)
df_min_max = pd.read_csv(BASE_DIR / '1_df_min_max.csv')
df_min = df_min_max['min'].values
df_max = df_min_max['max'].values

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
    results = {}

    with open(BASE_DIR / "./dtree.pkl", "rb") as f: # rb - режим чтения, ./model.pkl - файл куда сохраняется модель
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

        results["Decision Tree"] = {'Prediction': int(pred_tree), 
                                    'Probability No Default': round(float(p_no_default), 4), 
                                    'Probability Default': round(float(p_default), 4)}
    
    with open(BASE_DIR / "./rfc.pkl", "rb") as f: # rb - режим чтения, ./model.pkl - файл куда сохраняется модель
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

        results["Random Forest"] = {'Prediction': int(pred_rfc), 
                                    'Probability No Default': round(float(p_no_default), 4), 
                                    'Probability Default': round(float(p_default), 4)}

    with open(BASE_DIR / "./boost_gr.pkl", "rb") as f: # rb - режим чтения, ./model.pkl - файл куда сохраняется модель
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

        results["Gradient Boosting"] = {'Prediction': int(pred_boost_gr), 
                                        'Probability No Default': round(float(p_no_default), 4), 
                                        'Probability Default': round(float(p_default), 4)}

    model = keras.models.load_model(BASE_DIR / "my_model_3mln.h5", compile=False) 
   
    exp_nn = np.array(exp).reshape(1, 19)
    pred_proba_nn = model.predict([exp_nn])[0][0]

    p_default_nn = pred_proba_nn
    p_no_default_nn = 1 - pred_proba_nn

    if  p_no_default_nn > shold:
        pred_nn = 0
        st.write("Neural Network")
        st.success("0 -- no default")
        st.success(round(p_no_default_nn, 4))

    else:  
        pred_nn = 1  
        st.write("Neural Network")
        st.error("1 -- default")
        st.error(round(p_default_nn, 4))

    results["Neural Network"] = {'Prediction': int(pred_nn),
                                 'Probability No Default': round(float(p_no_default_nn), 4),
                                 'Probability Default': round(float(p_default_nn), 4)}

    features = ['enc_paym_0', 'enc_paym_1', 'enc_paym_2',
                  'enc_paym_3', 'enc_paym_4', 'enc_paym_5', 
                  'enc_paym_6', 'enc_paym_7', 'enc_paym_8', 
                  'enc_paym_9', 'enc_paym_10', 'enc_paym_11', 
                  'enc_paym_12', 'enc_paym_13', 'enc_paym_14', 
                  'enc_paym_15', 'enc_paym_22', 'enc_paym_23', 'enc_paym_24']

    explain_rfc = pd.DataFrame([exp], columns=features)
    explainer = shap.TreeExplainer(rfc) 
    shap_values = explainer(explain_rfc) 

    shap_default = shap_values.values[0, :, 1]
    shap_df = pd.DataFrame({'Feature': explain_rfc.columns,
                            'Value': explain_rfc.iloc[0].values,
                            'SHAP': shap_default})

    shap_df['Month'] = shap_df['Feature'].str.replace('enc_paym_','Месяц ',regex=False)

    risk_def = shap_df[shap_df["SHAP"] > 0].sort_values(by="SHAP", ascending=False)
    risk_nodef = shap_df[shap_df["SHAP"] < 0].sort_values(by="SHAP", ascending=True)

    top_risk = (risk_def[['Month', 'Value', 'SHAP']].head(3).to_dict(orient='records'))
    top_safe = (risk_nodef[['Month', 'Value', 'SHAP']].head(3).to_dict(orient='records'))

    llm_data = {"Результаты моделей": results,
                "Месяцы, увеличивающие вероятность дефолта": top_risk,
                "Месяцы, уменьшающие вероятность дефолта": top_safe}

    results_json = json.dumps(llm_data,           # превращение словаря в json-строку
                              ensure_ascii=False,
                              indent=2)
    
    response = llm_client.chat.completions.create(
    model="nvidia/nemotron-3.5-lightning:free",
    messages=[{"role": "system",
               "content": """
Ты помогаешь интерпретировать результаты
моделей кредитного скоринга.

Класс 0 означает no default.
Класс 1 означает default.

Отвечай только на русском языке.
Анализируй только предоставленные результаты моделей. 
Не придумывай дополнительные сведения о клиенте.
Не изменяй рассчитанные моделями вероятности.
На основе этих результатов кратко объясни, как модели 
оценивают кредитный риск и согласуются ли их результаты между собой.
"""},

{"role": "user",
 "content": f""" Результаты моделей: {results_json} 
Кратко объясни:
1. общий результат четырех моделей;
2. какие месяцы наиболее сильно повысили вероятность дефолта
   по модели Random Forest;
3. какие месяцы наиболее сильно снизили вероятность дефолта
   по модели Random Forest;
4. цельный итоговый вывод.
"""}])

    llm_answer = response.choices[0].message.content
    st.subheader("Интерпретация результатов с помощью LLM модели")
    st.write(llm_answer)