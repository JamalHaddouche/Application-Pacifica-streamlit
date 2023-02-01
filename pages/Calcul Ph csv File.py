import streamlit as st
import pandas as pd
import pickle
import numpy as np
##Model de machine learning
#5variables
_5var = pickle.load(open("model/allFeatures/pacifica_all_features_SVR.pkl", "rb"))

#4 variables
PHSPHT_SILICAT_OXYGENE_TMP=pickle.load(open("model/comb4v/SVR_4v_PHSPHT_SILCAT_OXYGEN_TMP_", "rb"))
PHSPHT_SILICAT_TCARBN_OXYGEN=pickle.load(open("model/comb4v/SVR_4v_PHSPHT_SILCAT_TCARBN_OXYGEN_", "rb"))
PHSPHT_SILICAT_TCARBN_TMP=pickle.load(open("model/comb4v/SVR_4v_PHSPHT_SILCAT_TCARBN_TMP_", "rb"))
PHSPHT_TCARBN_OXYGEN_TMP=pickle.load(open("model/comb4v/SVR_4v_PHSPHT_TCARBN_OXYGEN_TMP_", "rb"))
SILCAT_TCARBN_OXYGEN_TMP=pickle.load(open("model/comb4v/SVR_4v_SILCAT_TCARBN_OXYGEN_TMP_", "rb"))

#3 VARIABLES

PHSPHT_OXYGEN_TMP=pickle.load(open("model/comb3v/SVR_3v_PHSPHT_OXYGEN_TMP_", "rb"))
PHSPHT_SILCAT_OXYGEN=pickle.load(open("model/comb3v/SVR_3v_PHSPHT_SILCAT_OXYGEN_", "rb"))
PHSPHT_SILCAT_TCARBN=pickle.load(open("model/comb3v/SVR_3v_PHSPHT_SILCAT_TCARBN_", "rb"))
PHSPHT_SILCAT_TMP=pickle.load(open("model/comb3v/SVR_3v_PHSPHT_SILCAT_TMP_", "rb"))
PHSPHT_TCARBN_OXYGEN=pickle.load(open("model/comb3v/SVR_3v_PHSPHT_TCARBN_OXYGEN_", "rb"))
PHSPHT_TCARBN_TMP=pickle.load(open("model/comb3v/SVR_3v_PHSPHT_TCARBN_TMP_", "rb"))
SILCAT_OXYGEN_TMP=pickle.load(open("model/comb3v/SVR_3v_SILCAT_OXYGEN_TMP_", "rb"))
SILCAT_TCARBN_OXYGEN=pickle.load(open("model/comb3v/SVR_3v_SILCAT_TCARBN_OXYGEN_", "rb"))
SILCAT_TCARBN_TMP=pickle.load(open("model/comb3v/SVR_3v_SILCAT_TCARBN_TMP_", "rb"))
TCARBN_OXYGEN_TMP=pickle.load(open("model/comb3v/SVR_3v_TCARBN_OXYGEN_TMP_", "rb"))

#2 VARIABLES

OXYGEN_TMP=pickle.load(open("model/comb2v/SVR_2v_OXYGEN_TMP_","rb"))
PHSPHT_OXYGEN=pickle.load(open("model/comb2v/SVR_2v_PHSPHT_OXYGEN_","rb"))
PHSPHT_SILCAT=pickle.load(open("model/comb2v/SVR_2v_PHSPHT_SILCAT_","rb"))
PHSPHT_TCARBN=pickle.load(open("model/comb2v/SVR_2v_PHSPHT_TCARBN_","rb"))
PHSPHT_TMP=pickle.load(open("model/comb2v/SVR_2v_PHSPHT_TMP_","rb"))
SILCAT_OXYGEN=pickle.load(open("model/comb2v/SVR_2v_SILCAT_OXYGEN_","rb"))
SILCAT_TCARBN=pickle.load(open("model/comb2v/SVR_2v_SILCAT_TCARBN_","rb"))
SILCAT_TMP=pickle.load(open("model/comb2v/SVR_2v_SILCAT_TMP_","rb"))
TCARBN_OXYGEN=pickle.load(open("model/comb2v/SVR_2v_TCARBN_OXYGEN_","rb"))
TCARBN_TMP=pickle.load(open("model/comb2v/SVR_2v_TCARBN_TMP_","rb"))

#1 VARIABLE
TMP_model=pickle.load(open("model/oneFeature/SVR_1v_TMP_","rb"))
OXYGENE_model=pickle.load(open("model/oneFeature/SVR_1v_OXYGEN_","rb"))
PHOSPHT_model=pickle.load(open("model/oneFeature/SVR_1v_PHSPHT_","rb"))
SILICAT_model=pickle.load(open("model/oneFeature/SVR_1v_SILCAT_","rb"))
TCARBN_model=pickle.load(open("model/oneFeature/SVR_1v_TCARBN_","rb"))


st.title('Acidité océonique')

data= st.file_uploader('S\'il vous plait respecter l\'entete suivante:Température => TMP,Oxygène=> OXYGEN,Totale Carbonique=> TCARBN,Silicate=>SILCAT,Phosphate=> PHSPHT')
if data is not None:
    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(data)
    st.write('Choisir les variables a utilisé pour predire la valeur de PH')
    dataframe = dataframe[:20]
    Ph_Mesured = dataframe.filter(items=['PH'])
    datacopy=dataframe.copy()
    col1, col2, col3 = st.columns(3)
    nbVr = []
    prediction =[]
    with col1:
        if st.checkbox('Température'):
            nbVr.append(1)
        if st.checkbox('Silicate'):
            nbVr.append(2)
    with col2:
        if st.checkbox('Oxygène'):
            nbVr.append(3)
        if st.checkbox('Totale Carbonique'):
            nbVr.append(4)
    with col3:
        if st.checkbox('Phosphate'):
            nbVr.append(5)
    if (st.button('Estimer les valeur PH') and len(nbVr) != 0):
        ##si l'utilasateur a choisi une seul variable
        if (len(nbVr) == 1):
            if (nbVr[0] == 1):
                Features = dataframe.filter(items=['TMP'])
                prediction = TMP_model.predict(Features)
            elif (nbVr[0] == 2):
                Features = dataframe.filter(items=['SILCAT'])
                prediction = SILICAT_model.predict(Features)
            elif (nbVr[0] == 3):
                Features = dataframe.filter(items=['OXYGEN'])
                prediction = OXYGENE_model.predict(Features)

            elif (nbVr[0] == 4):
                Features = dataframe.filter(items=['TCARBN'])
                prediction = TCARBN_model.predict(Features)
            elif (nbVr[0] == 5):
                Features = dataframe.filter(items=['PHSPHT'])
                prediction = PHOSPHT_model.predict(Features)

        # Deux variables
        if len(nbVr) == 2:
            if (nbVr == [1, 2]):
                Features = dataframe.filter(items=['SILCAT','TMP'])
                prediction = SILCAT_TMP.predict(Features)

            elif (nbVr == [1, 3]):
                Features = dataframe.filter(items=['OXYGEN','TMP'])
                prediction = OXYGEN_TMP.predict(Features)

            elif (nbVr == [1, 4]):
                Features = dataframe.filter(items=['TCARBN','TMP'])
                prediction = TCARBN_TMP.predict(Features)

            elif (nbVr == [2, 4]):
                Features = dataframe.filter(items=['SILCAT','TCARBN'])
                prediction = SILCAT_TCARBN.predict(Features)

            elif (nbVr == [2, 3]):
                Features = dataframe.filter(items=['SILCAT','OXYGEN'])
                prediction = SILCAT_OXYGEN.predict(Features)

            elif (nbVr == [3, 4]):
                Features = dataframe.filter(items=['TCARBN','OXYGEN'])
                prediction = TCARBN_OXYGEN.predict(Features)

            elif (nbVr == [1, 5]):
                Features = dataframe.filter(items=['PHSPHT','TMP'])
                prediction = PHSPHT_TMP.predict(Features)

            elif (nbVr == [2, 5]):
                Features = dataframe.filter(items=['PHSPHT', 'SILCAT'])
                prediction = PHSPHT_SILCAT.predict(Features)

            elif (nbVr == [3, 5]):
                Features = dataframe.filter(items=['PHSPHT', 'OXYGEN'])
                prediction = PHSPHT_OXYGEN.predict(Features)

            elif (nbVr == [4, 5]):
                Features = dataframe.filter(items=['PHSPHT', 'TCARBN'])
                prediction = PHSPHT_TCARBN.predict(Features)

        # 3 variables
        if len(nbVr) == 3:
            if (nbVr == [1, 3, 5]):
                Features = dataframe.filter(items=['PHSPHT', 'OXYGEN','TMP'])
                prediction = PHSPHT_OXYGEN_TMP.predict(Features)
            elif (nbVr == [2, 3, 5]):
                Features = dataframe.filter(items=['PHSPHT', 'SILCAT', 'OXYGEN'])
                prediction = PHSPHT_SILCAT_OXYGEN.predict(Features)
            elif (nbVr == [2, 4, 5]):
                Features = dataframe.filter(items=['PHSPHT', 'SILCAT', 'TCARBN'])
                prediction = PHSPHT_SILCAT_TCARBN.predict(Features)
            elif (nbVr == [1, 2, 5]):
                Features = dataframe.filter(items=['PHSPHT', 'SILCAT', 'TMP'])
                prediction = PHSPHT_SILCAT_TMP.predict(Features)
            elif (nbVr == [3, 4, 5]):
                Features = dataframe.filter(items=['PHSPHT', 'TCARBN', 'OXYGEN'])
                prediction = PHSPHT_TCARBN_OXYGEN.predict(Features)
            elif (nbVr == [1, 4, 5]):
                Features = dataframe.filter(items=['PHSPHT', 'TCARBN', 'TMP'])
                prediction = PHSPHT_TCARBN_TMP.predict(Features)
            elif (nbVr == [1, 2, 3]):
                Features = dataframe.filter(items=['SILCAT', 'OXYGEN', 'TMP'])
                prediction = SILCAT_OXYGEN_TMP.predict(Features)
            elif (nbVr == [2, 3, 4]):
                Features = dataframe.filter(items=['SILCAT', 'TCARBN', 'OXYGEN'])
                prediction = SILCAT_TCARBN_OXYGEN.predict(Features)
            elif (nbVr == [1, 2, 4]):
                Features = dataframe.filter(items=['SILCAT', 'TCARBN', 'TMP'])
                prediction = SILCAT_TCARBN_TMP.predict(Features)
            elif (nbVr == [1, 3, 4]):
                Features = dataframe.filter(items=['TCARBN', 'OXYGEN', 'TMP'])
                prediction = TCARBN_OXYGEN_TMP.predict(Features)
        # 4 VARIABLES
        if len(nbVr) == 4:
            if (nbVr == [1, 2, 3, 5]):
                df = dataframe.filter(items=['PHSPHT', 'SILCAT', 'OXYGEN','TMP'])
                prediction = PHSPHT_SILICAT_OXYGENE_TMP.predict(df)
            elif (nbVr == [2, 3, 4, 5]):
                df = dataframe.filter(items=['PHSPHT', 'SILCAT', 'TCARBN', 'OXYGEN'])
                prediction = PHSPHT_SILICAT_TCARBN_OXYGEN.predict(df)
            elif (nbVr == [1, 2, 4, 5]):
                df = dataframe.filter(items=['PHSPHT', 'SILCAT', 'TCARBN', 'TMP'])
                prediction = PHSPHT_SILICAT_TCARBN_TMP.predict(df)
            elif (nbVr == [1, 3, 4, 5]):
                df = dataframe.filter(items=['PHSPHT', 'TCARBN', 'OXYGEN', 'TMP'])
                prediction = PHSPHT_TCARBN_OXYGEN_TMP.predict(df)
            elif (nbVr == [1, 2, 3, 4]):
                df = dataframe.filter(items=['SILCAT', 'TCARBN', 'OXYGEN', 'TMP'])
                prediction = SILCAT_TCARBN_OXYGEN_TMP.predict(df)
        # 5 VARIABLES
        if len(nbVr) == 5:
            df = dataframe.filter(items=['TMP','OXYGEN','TCARBN','SILCAT', 'PHSPHT'])
            prediction = _5var.predict(df)


        Ph_Mesured['PH_Predicted']=prediction
        st.line_chart(Ph_Mesured)

        #Download les données avec leurs Ph estimer
        data_download = dataframe.filter(items=['TMP', 'OXYGEN', 'TCARBN', 'SILCAT', 'PHSPHT'])
        data_download['Ph_Predicte']=prediction
        @st.cache
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')


        csv = convert_df(data_download)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='EstmationPh.csv',
            mime='text/csv',
        )
