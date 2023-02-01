import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import pandas as pd
import pickle


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


#reshape des valeurs saisie par l'utilisateur
def features(float_features):
    ft = list(map(float, float_features))
    fts = np.array(ft)
    fts = fts.reshape(1, -1)
    return fts


img1 = Image.open('img\img1.png')
img2 = Image.open('img\pacificaCorrelation.png')
#Titre de projet et son but
st.title('Acidité Océonique ')
st.image(img1)
st.write('Le but de ce projet et d\'estimer la valeur de ph utilisant des différents  modeèls de machine learning et des différents variables .')


#Container Dataset presentation
dt=st.expander("Base de données utilisé")
pacifica=pd.read_csv(r'dataset/PacificaClean.csv')
pacifica=pacifica.drop(columns=['Unnamed: 0'])
dt.text('exemple de 5 premier ligne de notre dataset après le nettoyage')
dt.dataframe(pacifica.head(5))
dt.text('matrice de corrélation')
dt.image(img2)
dt.write("✅ On remarque d'après la matrice de corrèlation que les varables suivants qui influencent plus dans la variation de ph")
#Container Machine learning acidité
#Container Machine learning acidité
col1, col2, col3 = st.columns(3)
nbVr=[]
prediction=0
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
            float_features = [tmp]
            prediction = TMP_model.predict(features(float_features))
        elif (nbVr[0] == 2):
            float_features = [slc]
            prediction = SILICAT_model.predict(features(float_features))

        elif (nbVr[0] ==3):
            float_features = [oxy]
            prediction = OXYGENE_model.predict(features(float_features))

        elif (nbVr[0] == 4):
            float_features = [tCarbn]
            prediction = TCARBN_model.predict(features(float_features))
        elif (nbVr[0] == 5):
            float_features = [pho]
            prediction = PHOSPHT_model.predict(features(float_features))

    #Deux variables
    if len(nbVr) == 2:
        if (nbVr == [1, 2]):
            df = pd.DataFrame(data={'SILCAT': slc, 'TMP': tmp},index=[0])
            df = np.array(df)
            print(df)
            prediction = SILCAT_TMP.predict(df)

        elif (nbVr == [1, 3]):
            df = pd.DataFrame(data={'OXYGEN': oxy, 'TMP': tmp}, index=[0])
            df = np.array(df)
            prediction = OXYGEN_TMP.predict(df)

        elif (nbVr == [1,4]):
            df = pd.DataFrame(data={'TCARBN': tCarbn, 'TMP': tmp}, index=[0])
            df = np.array(df)
            prediction = TCARBN_TMP.predict(df)

        elif (nbVr == [2, 4]):
            df = pd.DataFrame(data={'SILCAT':slc,'TCARBN': tCarbn}, index=[0])
            df = np.array(df)
            prediction = SILCAT_TCARBN.predict(df)

        elif (nbVr == [2, 3]):
            df = pd.DataFrame(data={'SILCAT': slc, 'OXYGEN': oxy}, index=[0])
            df = np.array(df)
            prediction = SILCAT_OXYGEN.predict(df)

        elif (nbVr == [3, 4]):
            df = pd.DataFrame(data={'TCARBN': tCarbn, 'OXYGEN': oxy}, index=[0])
            df = np.array(df)
            prediction = TCARBN_OXYGEN.predict(df)
        elif (nbVr == [1, 5]):
            df = pd.DataFrame(data={'PHSPHT': pho, 'TMP': tmp}, index=[0])
            df = np.array(df)
            prediction = PHSPHT_TMP.predict(df)
        elif (nbVr == [2, 5]):
            df = pd.DataFrame(data={'PHSPHT': pho, 'SILCAT': slc}, index=[0])
            df = np.array(df)
            prediction = PHSPHT_SILCAT.predict(df)
        elif (nbVr == [3, 5]):
            df = pd.DataFrame(data={'PHSPHT': pho, 'OXYGEN': oxy}, index=[0])
            df = np.array(df)
            prediction = PHSPHT_OXYGEN.predict(df)
        elif (nbVr == [4 , 5]):
            df = pd.DataFrame(data={'PHSPHT': pho, 'TCARBN': tCarbn}, index=[0])
            df = np.array(df)
            prediction = PHSPHT_TCARBN.predict(df)
    #3 variables
    if len(nbVr)==3:
        if (nbVr == [1,3,5]):
            df = pd.DataFrame(data={'PHSPHT': pho, 'OXYGEN': oxy,'TMP':tmp}, index=[0])
            df = np.array(df)
            prediction = PHSPHT_OXYGEN_TMP.predict(df)
        elif (nbVr == [2,3,5]):
            df = pd.DataFrame(data={'PHSPHT': pho, 'SILCAT': slc, 'OXYGEN': oxy}, index=[0])
            df = np.array(df)
            prediction = PHSPHT_SILCAT_OXYGEN.predict(df)
        elif (nbVr == [2,4,5]):
            df = pd.DataFrame(data={'PHSPHT': pho, 'SILCAT': slc, 'TCARBN': tCarbn}, index=[0])
            df = np.array(df)
            prediction = PHSPHT_SILCAT_TCARBN.predict(df)
        elif (nbVr == [1,2,5]):
            df = pd.DataFrame(data={'PHSPHT': pho, 'SILCAT': slc, 'TMP':tmp}, index=[0])
            df = np.array(df)
            prediction = PHSPHT_SILCAT_TMP.predict(df)
        elif (nbVr == [3,4,5]):
            df = pd.DataFrame(data={'PHSPHT': pho, 'TCARBN': tCarbn,'OXYGEN': oxy}, index=[0])
            df = np.array(df)
            prediction = PHSPHT_TCARBN_OXYGEN.predict(df)
        elif (nbVr == [1,4,5]):
            df = pd.DataFrame(data={'PHSPHT': pho, 'TCARBN': tCarbn,'TMP':tmp}, index=[0])
            df = np.array(df)
            prediction = PHSPHT_TCARBN_TMP.predict(df)
        elif (nbVr == [1,2,3]):
            df = pd.DataFrame(data={'SILCAT': slc, 'OXYGEN': oxy,'TMP':tmp}, index=[0])
            df = np.array(df)
            prediction = SILCAT_OXYGEN_TMP.predict(df)
        elif (nbVr == [2,3,4]):
            df = pd.DataFrame(data={'SILCAT': slc, 'TCARBN': tCarbn,'OXYGEN': oxy}, index=[0])
            df = np.array(df)
            prediction = SILCAT_TCARBN_OXYGEN.predict(df)
        elif (nbVr == [1,2,4]):
            df = pd.DataFrame(data={'SILCAT': slc, 'TCARBN': tCarbn, 'TMP':tmp}, index=[0])
            df = np.array(df)
            prediction = SILCAT_TCARBN_TMP.predict(df)
        elif (nbVr == [1,3,4]):
            df = pd.DataFrame(data={'TCARBN': tCarbn,'OXYGEN': oxy, 'TMP': tmp}, index=[0])
            df = np.array(df)
            prediction = TCARBN_OXYGEN_TMP.predict(df)
    # 4 VARIABLES
    if len(nbVr)==4:
        if (nbVr == [1,2,3,5]):
            df = pd.DataFrame(data={'PHSPHT': pho,'SILCAT': slc,'OXYGEN': oxy, 'TMP': tmp}, index=[0])
            df = np.array(df)
            prediction = PHSPHT_SILICAT_OXYGENE_TMP.predict(df)
        elif(nbVr == [2,3,4,5]):
            df = pd.DataFrame(data={'PHSPHT': pho,'SILCAT': slc,'TCARBN': tCarbn, 'OXYGEN': oxy}, index=[0])
            df = np.array(df)
            prediction = PHSPHT_SILICAT_TCARBN_OXYGEN.predict(df)
        elif (nbVr == [1,2,4,5]):
            df = pd.DataFrame(data={'PHSPHT': pho, 'SILCAT': slc, 'TCARBN': tCarbn,'TMP': tmp}, index=[0])
            df = np.array(df)
            prediction = PHSPHT_SILICAT_TCARBN_TMP.predict(df)
        elif (nbVr == [1,3,4,5]):
            df = pd.DataFrame(data={'PHSPHT': pho,'TCARBN': tCarbn,'OXYGEN': oxy,'TMP': tmp}, index=[0])
            df = np.array(df)
            prediction = PHSPHT_TCARBN_OXYGEN_TMP.predict(df)
        elif (nbVr == [1,2,3,4]):
            df = pd.DataFrame(data={'SILCAT': slc,'TCARBN': tCarbn,'OXYGEN': oxy,'TMP': tmp}, index=[0])
            df=np.array(df)
            prediction = SILCAT_TCARBN_OXYGEN_TMP.predict(df)
    #5 VARIABLES
    if len(nbVr)==5:
        float_features = [tmp,oxy,tCarbn,slc,pho]
        prediction = _5var.predict([float_features])
    print("###########",nbVr)
    print(prediction)
    st.success(prediction)

