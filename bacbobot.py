import cv2
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
from mss import mss
import matplotlib.pyplot as plt
import plotly.express as px
import torch
import torchvision.transforms as transforms
from PIL import Image

# Configuração do Tesseract (ajuste o caminho conforme necessário)
pytesseract.pytesseract.tesseract_cmd = r'C:\Arquivos de Programas\Tesseract-OCR\tesseract.exe'  # Exemplo para Windows

# Configurações para captura de tela
monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}  # Ajuste conforme sua tela
sct = mss()

# Função para capturar a tela
def capturar_tela():
    sct_img = sct.grab(monitor)
    frame = np.array(sct_img)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame

# Função para processar a imagem e identificar os números
def processar_imagem(frame):
    # Converta para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplique um filtro para destacar os números
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Encontre contornos na imagem
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    numeros = []
    for contorno in contornos:
        # Filtre contornos pequenos (ruído)
        if cv2.contourArea(contorno) > 100:
            (x, y, w, h) = cv2.boundingRect(contorno)
            roi = gray[y:y + h, x:x + w]  # Região de interesse (número)
            texto = pytesseract.image_to_string(roi, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
            if texto.strip().isdigit():
                numeros.append({"x": x, "y": y, "numero": int(texto.strip())})

    # Ordenar números pela posição (esquerda para direita)
    numeros.sort(key=lambda n: n["x"])
    return numeros

# Função para calcular probabilidades
def calcular_probabilidades(historico):
    df = pd.DataFrame(historico)
    prob_player = df["player"].value_counts(normalize=True) * 100
    prob_banker = df["banker"].value_counts(normalize=True) * 100
    return prob_player, prob_banker

# Função para exibir gráficos
def exibir_graficos(historico):
    df = pd.DataFrame(historico)
    st.write("### Gráfico de Tendências")
    fig = px.line(df, x=df.index, y=["player", "banker"], title="Tendências Player vs Banker")
    st.plotly_chart(fig)

# Função para calcular estatísticas
def calcular_estatisticas(historico):
    df = pd.DataFrame(historico)
    estatisticas = {
        "Média Player": df["player"].mean(),
        "Desvio Padrão Player": df["player"].std(),
        "Média Banker": df["banker"].mean(),
        "Desvio Padrão Banker": df["banker"].std()
    }
    return estatisticas

# Função principal
def analisar_tela_ao_vivo():
    st.title("Análise de Bac-Bo em Tempo Real 🎲")
    st.write("Capturando e analisando a tela ao vivo...")

    # Inicializar histórico
    if "historico" not in st.session_state:
        st.session_state.historico = []

    # Capturar e processar a tela em tempo real
    frame = capturar_tela()
    numeros = processar_imagem(frame)

    if len(numeros) >= 2:  # Pelo menos dois números (Player e Banker)
        player = numeros[0]["numero"]
        banker = numeros[1]["numero"]
        st.write(f"Player: {player}, Banker: {banker}")

        # Atualizar histórico
        st.session_state.historico.append({"player": player, "banker": banker})

        # Calcular probabilidades históricas
        prob_player, prob_banker = calcular_probabilidades(st.session_state.historico)
        st.write("Probabilidades Player:")
        st.write(prob_player)
        st.write("Probabilidades Banker:")
        st.write(prob_banker)

        # Exibir gráficos
        exibir_graficos(st.session_state.historico)

        # Calcular estatísticas
        estatisticas = calcular_estatisticas(st.session_state.historico)
        st.write("### Estatísticas")
        st.write(estatisticas)

# Executar o bot
if __name__ == "__main__":
    analisar_tela_ao_vivo()
