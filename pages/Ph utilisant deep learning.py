import folium as fl
from streamlit_folium import st_folium
import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model

#Titre de projet et son but
st.title('Estimation de la valeur de ph ')

#reshape des valeurs saisie par l'utilisateur
def features(float_features):
    ft = list(map(float, float_features))
    fts = np.array(ft)
    fts = fts.reshape(1, -1)
    return fts


def get_pos(lat,lng):
    return lat,lng

m = fl.Map()
m.add_child(fl.LatLngPopup())
map = st_folium(m, height=350, width=700)
if map['last_clicked'] is not None:
    data = get_pos(map['last_clicked']['lat'],map['last_clicked']['lng'])

##################

nbVr=[]
prediction=0

depth = st.slider('Depth', 0, 7000, 25)

col1, col2, col3 = st.columns(3)
with col1:
    if st.checkbox('Température'):
        nbVr.append(1)
        tmp = st.number_input('Température')
    if st.checkbox('Silicate'):
        nbVr.append(2)
        slc = st.number_input('Silicate')
with col2:

    if st.checkbox('Oxygène'):
        nbVr.append(3)
        oxy = st.number_input('Oxygène')
    if st.checkbox('Totale Carbonique'):
        nbVr.append(4)
        tCarbn = st.number_input('TCARBN')
with col3:

    if st.checkbox('Phosphate'):
        nbVr.append(5)
        pho = st.number_input('Phosphate')
#ml= st.expander("Prédire la valeur de PH avec Machine learning")

    if st.button("Estimer la valeur de ph"):
        ##si l'utilasateur a choisi une seul variable
        if (len(nbVr) == 1):
            if (nbVr[0] == 1):
                TEMPERATURE = load_model('model/AnnModelExperience/TMP_.h5')
                float_features = [data[0],data[1],depth,tmp]
                prediction = TEMPERATURE.predict([float_features])
            elif (nbVr[0] == 2):
                SILICAT = load_model('model/AnnModelExperience/SILCAT_.h5')
                float_features = [data[0],data[1],depth,slc]
                prediction = SILICAT.predict([float_features])

            elif (nbVr[0] ==3):
                OXYGEN = load_model('model/AnnModelExperience/OXYGEN_.h5')
                float_features = [data[0],data[1],depth,oxy]
                prediction = OXYGEN.predict([float_features])

            elif (nbVr[0] == 4):
                TCARBN = load_model('model/AnnModelExperience/TCARBN_.h5')
                float_features = [data[0],data[1],depth,tCarbn]
                prediction = TCARBN.predict([float_features])
            elif (nbVr[0] == 5):
                PHSPHT_ = load_model('model/AnnModelExperience/PHSPHT_.h5')
                float_features = [data[0],data[1],depth,pho]
                prediction = PHSPHT_.predict([float_features])

        #Deux variables
        if len(nbVr) == 2:
            if (nbVr == [1, 2]):
                SILCAT_TMP = load_model('model/AnnModelExperience/SILCAT_TMP_.h5')
                float_features = [data[0], data[1], depth, slc,tmp]
                prediction = SILCAT_TMP.predict([float_features])

            elif (nbVr == [1, 3]):
                OXYGEN_TMP = load_model('model/AnnModelExperience/OXYGEN_TMP_.h5')
                float_features = [data[0], data[1], depth, oxy, tmp]
                prediction = OXYGEN_TMP.predict([float_features])
            elif (nbVr == [1,4]):
                TCARBN_TMP = load_model('model/AnnModelExperience/TCARBN_TMP_.h5')
                float_features = [data[0], data[1], depth, tCarbn, tmp]
                prediction = TCARBN_TMP.predict([float_features])

            elif (nbVr == [2, 4]):
                SILCAT_TCARBN = load_model('model/AnnModelExperience/SILCAT_TCARBN_.h5')
                float_features = [data[0], data[1], depth,slc, tCarbn]
                prediction = SILCAT_TCARBN.predict([float_features])

            elif (nbVr == [2, 3]):
                SILCAT_OXYGEN = load_model('model/AnnModelExperience/SILCAT_OXYGEN_.h5')
                df = pd.DataFrame(data={'Longitude':data[0],'Latitude':data[1],'DEPTH':depth,'SILCAT': slc, 'OXYGEN': oxy}, index=[0])
                prediction = SILCAT_OXYGEN.predict(df)

            elif (nbVr == [3, 4]):
                TCARBN_OXYGEN = load_model('model/AnnModelExperience/TCARBN_OXYGEN_.h5')
                float_features = [data[0], data[1], depth, tCarbn, tmp]
                df = pd.DataFrame(data={'Longitude':data[0],'Latitude':data[1],'DEPTH':depth,'TCARBN': tCarbn, 'OXYGEN': oxy}, index=[0])
                prediction = TCARBN_OXYGEN.predict(df)
            elif (nbVr == [1, 5]):
                PHSPHT_TMP = load_model('model/AnnModelExperience/PHSPHT_TMP_.h5')
                df = pd.DataFrame(data={'Longitude':data[0],'Latitude':data[1],'DEPTH':depth,'PHSPHT': pho, 'TMP': tmp}, index=[0])
                prediction = PHSPHT_TMP.predict(df)
            elif (nbVr == [2, 5]):
                PHSPHT_SILCAT = load_model('model/AnnModelExperience/PHSPHT_SILCAT_.h5')
                df = pd.DataFrame(data={'Longitude':data[0],'Latitude':data[1],'DEPTH':depth,'PHSPHT': pho, 'SILCAT': slc}, index=[0])
                prediction = PHSPHT_SILCAT.predict(df)
            elif (nbVr == [3, 5]):
                PHSPHT_OXYGEN = load_model('model/AnnModelExperience/PHSPHT_OXYGEN_.h5')
                df = pd.DataFrame(data={'Longitude':data[0],'Latitude':data[1],'DEPTH':depth,'PHSPHT': pho, 'OXYGEN': oxy}, index=[0])
                prediction = PHSPHT_OXYGEN.predict(df)
            elif (nbVr == [4 , 5]):
                PHSPHT_TCARBN = load_model('model/AnnModelExperience/PHSPHT_TCARBN_.h5')
                df = pd.DataFrame(data={'Longitude':data[0],'Latitude':data[1],'DEPTH':depth,'PHSPHT': pho, 'TCARBN': tCarbn}, index=[0])
                prediction = PHSPHT_TCARBN.predict(df)
        #3 variables
        if len(nbVr)==3:
            if (nbVr == [1,3,5]):
                PHSPHT_OXYGEN_TMP = load_model('model/AnnModelExperience/PHSPHT_OXYGEN_TMP_.h5')
                df = pd.DataFrame(data={'Longitude':data[0],'Latitude':data[1],'DEPTH':depth,'PHSPHT': pho, 'OXYGEN': oxy,'TMP':tmp}, index=[0])
                df=np.array(df)
                prediction = PHSPHT_OXYGEN_TMP.predict(df)
            elif (nbVr == [2,3,5]):
                PHSPHT_SILCAT_OXYGEN = load_model('model/AnnModelExperience/PHSPHT_SILCAT_OXYGEN_.h5')
                df = pd.DataFrame(data={'Longitude':data[0],'Latitude':data[1],'DEPTH':depth,'PHSPHT': pho, 'SILCAT': slc, 'OXYGEN': oxy}, index=[0])
                df = np.array(df)
                print(df)
                prediction = PHSPHT_SILCAT_OXYGEN.predict(df)
            elif (nbVr == [2,4,5]):
                PHSPHT_SILCAT_TCARBN = load_model('model/AnnModelExperience/PHSPHT_SILCAT_TCARBN_.h5')
                df = pd.DataFrame(data={'Longitude':data[0],'Latitude':data[1],'DEPTH':depth,'PHSPHT': pho, 'SILCAT': slc, 'OXYGEN': tCarbn}, index=[0])
                df = np.array(df)
                prediction = PHSPHT_SILCAT_TCARBN.predict(df)
            elif (nbVr == [1,2,5]):
                PHSPHT_SILCAT_TMP = load_model('model/AnnModelExperience/PHSPHT_SILCAT_TMP_.h5')
                df = pd.DataFrame(data={'Longitude':data[0],'Latitude':data[1],'DEPTH':depth,'PHSPHT': pho, 'SILCAT': slc, 'TMP':tmp}, index=[0])
                df = np.array(df)
                prediction = PHSPHT_SILCAT_TMP.predict(df)
            elif (nbVr == [3,4,5]):
                PHSPHT_TCARBN_OXYGEN = load_model('model/AnnModelExperience/PHSPHT_TCARBN_OXYGEN_.h5')
                df = pd.DataFrame(data={'Longitude':data[0],'Latitude':data[1],'DEPTH':depth,'PHSPHT': pho, 'TCARBN': tCarbn,'OXYGEN': oxy}, index=[0])
                df = np.array(df)
                prediction = PHSPHT_TCARBN_OXYGEN.predict(df)
            elif (nbVr == [1,4,5]):
                PHSPHT_TCARBN_TMP = load_model('model/AnnModelExperience/PHSPHT_TCARBN_TMP_.h5')
                df = pd.DataFrame(data={'Longitude':data[0],'Latitude':data[1],'DEPTH':depth,'PHSPHT': pho, 'TCARBN': tCarbn,'TMP':tmp}, index=[0])
                df = np.array(df)
                prediction = PHSPHT_TCARBN_TMP.predict(df)
            elif (nbVr == [1,2,3]):
                SILCAT_OXYGEN_TMP = load_model('model/AnnModelExperience/SILCAT_OXYGEN_TMP_.h5')
                df = pd.DataFrame(data={'Longitude':data[0],'Latitude':data[1],'DEPTH':depth,'SILCAT': slc, 'OXYGEN': oxy,'TMP':tmp}, index=[0])
                df = np.array(df)
                prediction = SILCAT_OXYGEN_TMP.predict(df)
            elif (nbVr == [2,3,4]):
                SILCAT_TCARBN_OXYGEN = load_model('model/AnnModelExperience/SILCAT_TCARBN_OXYGEN_.h5')
                df = pd.DataFrame(data={'Longitude':data[0],'Latitude':data[1],'DEPTH':depth,'SILCAT': slc, 'TCARBN': tCarbn,'OXYGEN': oxy}, index=[0])
                df = np.array(df)
                prediction = SILCAT_TCARBN_OXYGEN.predict(df)
            elif (nbVr == [1, 2, 4]):
                SILCAT_TCARBN_TMP = load_model('model/AnnModelExperience/SILCAT_TCARBN_TMP_.h5')
                #df = pd.DataFrame(data={'Longitude':data[0],'Latitude':data[1],'DEPTH':depth,'SILCAT': slc, 'TCARBN': tCarbn, 'TMP':tmp}, index=[0])
                df=[data[0],data[1],depth,slc,tCarbn,tmp]
                print(df)
                prediction = SILCAT_TCARBN_TMP.predict(df)
            elif (nbVr == [1,3,4]):
                TCARBN_OXYGEN_TMP = load_model('model/AnnModelExperience/TCARBN_OXYGEN_TMP_.h5')
                df = pd.DataFrame(data={'Longitude':data[0],'Latitude':data[1],'DEPTH':depth,'TCARBN': tCarbn,'OXYGEN': oxy, 'TMP': tmp}, index=[0])
                df = np.array(df)
                prediction = TCARBN_OXYGEN_TMP.predict(df)
        # 4 VARIABLES
        if len(nbVr)==4:
            if (nbVr == [1,2,3,5]):
                PHSPHT_SILCAT_OXYGEN_TMP = load_model('model/AnnModelExperience/PHSPHT_SILCAT_OXYGEN_TMP_.h5')
                df = pd.DataFrame(data={'Longitude':data[0],'Latitude':data[1],'DEPTH':depth,'PHSPHT': pho,'SILCAT': slc,'OXYGEN': oxy, 'TMP': tmp}, index=[0])
                df=np.array(df)
                prediction = PHSPHT_SILCAT_OXYGEN_TMP.predict(df)
            elif(nbVr == [2,3,4,5]):
                PHSPHT_SILCAT_TCARBN_OXYGEN = load_model('model/AnnModelExperience/PHSPHT_SILCAT_TCARBN_OXYGEN_.h5')
                df = pd.DataFrame(data={'Longitude':data[0],'Latitude':data[1],'DEPTH':depth,'PHSPHT': pho,'SILCAT': slc,'TCARBN': tCarbn, 'OXYGEN': oxy}, index=[0])
                df=np.array(df)
                prediction = PHSPHT_SILCAT_TCARBN_OXYGEN.predict(df)
            elif (nbVr == [1,2,4,5]):
                PHSPHT_SILCAT_TCARBN_TMP = load_model('model/AnnModelExperience/PHSPHT_SILCAT_TCARBN_TMP_.h5')
                df = pd.DataFrame(data={'Longitude':data[0],'Latitude':data[1],'DEPTH':depth,'PHSPHT': pho, 'SILCAT': slc, 'TCARBN': tCarbn,'TMP': tmp}, index=[0])
                df=np.array(df)
                prediction = PHSPHT_SILCAT_TCARBN_TMP.predict(df)
            elif (nbVr == [1,3,4,5]):
                PHSPHT_TCARBN_OXYGEN_TMP = load_model('model/AnnModelExperience/PHSPHT_TCARBN_OXYGEN_TMP_.h5')
                df = pd.DataFrame(data={'Longitude':data[0],'Latitude':data[1],'DEPTH':depth,'PHSPHT': pho,'TCARBN': tCarbn,'OXYGEN': oxy,'TMP': tmp}, index=[0])
                df=np.array(df)
                prediction = PHSPHT_TCARBN_OXYGEN_TMP.predict(df)
            elif (nbVr == [1,2,3,4]):
                SILCAT_TCARBN_OXYGEN_TMP = load_model('model/AnnModelExperience/SILCAT_TCARBN_OXYGEN_TMP_.h5')
                df = pd.DataFrame(data={'Longitude':data[0],'Latitude':data[1],'DEPTH':depth,'SILCAT': slc,'TCARBN': tCarbn,'OXYGEN': oxy,'TMP': tmp}, index=[0])
                df=np.array(df)
                prediction = SILCAT_TCARBN_OXYGEN_TMP.predict(df)
        #5 VARIABLES
        if len(nbVr)==5:
            allVariable = load_model('model/AnnModelExperience/experienceAnn.h5')
            float_features = [data[0],data[1],depth,tmp,oxy,tCarbn,slc,pho]
            prediction = allVariable.predict([float_features])
print("################",nbVr)
print(prediction)
st.success(prediction)

