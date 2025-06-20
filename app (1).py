
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# --- Simulando dataset ---
def gerar_dados(n=200):
    np.random.seed(42)
    temp = np.random.normal(loc=28, scale=5, size=n)
    hum = np.random.normal(loc=55, scale=10, size=n)
    lux = np.random.normal(loc=60, scale=20, size=n)
    irrigar = ((temp > 30) & (hum < 50)).astype(int)
    df = pd.DataFrame({'Temperatura': temp, 'Umidade': hum, 'Luminosidade': lux, 'Irrigar': irrigar})
    return df

df = gerar_dados()

X = df[['Temperatura', 'Umidade', 'Luminosidade']]
y = df['Irrigar']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(max_depth=4)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.title('FarmTech Solutions - Sistema de Irrigação Inteligente')
st.markdown(f"### Acurácia do modelo: {acc*100:.2f}%")

st.subheader('Dados de sensores simulados')
st.dataframe(df.head(20))

fig, ax = plt.subplots(1, 3, figsize=(15,4))
ax[0].hist(df['Temperatura'], bins=20, color='orange')
ax[0].set_title('Temperatura (°C)')
ax[1].hist(df['Umidade'], bins=20, color='blue')
ax[1].set_title('Umidade (%)')
ax[2].hist(df['Luminosidade'], bins=20, color='yellow')
ax[2].set_title('Luminosidade (lux)')
st.pyplot(fig)

st.subheader('Simule os dados do sensor')

temp_in = st.slider('Temperatura (°C)', 10, 50, 30)
hum_in = st.slider('Umidade (%)', 10, 90, 50)
lux_in = st.slider('Luminosidade (lux)', 0, 150, 60)

entrada = pd.DataFrame({'Temperatura':[temp_in],'Umidade':[hum_in],'Luminosidade':[lux_in]})
previsao = model.predict(entrada)[0]

st.markdown('### Resultado da previsão de irrigação:')

if previsao == 1:
    st.success('✅ Recomendado irrigar')
else:
    st.info('💧 Irrigação não necessária')

prob = model.predict_proba(entrada)[0][1]
st.write(f'Probabilidade de irrigar: {prob*100:.1f}%')
