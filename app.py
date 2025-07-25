import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("Previsão com Regressão Linear")
st.write("Digite os dados de entrada:")

x1 = st.number_input("Horas estudadas", min_value=0.0)
x2 = st.number_input("Horas dormidas", min_value=0.0)

df = pd.DataFrame({
    "horas_estudo": [1, 2, 3, 4, 5],
    "horas_sono": [8, 7, 6, 5, 4],
    "nota": [60, 65, 70, 75, 80]
})

X = df[["horas_estudo", "horas_sono"]]
y = df["nota"]

modelo = LinearRegression()
modelo.fit(X, y)

entrada = pd.DataFrame({"horas_estudo": [x1], "horas_sono": [x2]})
previsao = modelo.predict(entrada)[0]

st.write(f"Nota prevista: {previsao:.2f}")
