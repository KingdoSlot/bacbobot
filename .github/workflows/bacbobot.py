import logging
import os
import platform
import shutil

import cv2
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
from mss import mss
import plotly.express as px

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Configuração do Tesseract — detecta automaticamente ou usa variável de ambiente
_tesseract_cmd = os.environ.get("TESSERACT_CMD")
if _tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd
elif platform.system() == "Windows":
    _win_path = r"C:\Arquivos de Programas\Tesseract-OCR\tesseract.exe"
    if os.path.isfile(_win_path):
        pytesseract.pytesseract.tesseract_cmd = _win_path
    else:
        logger.warning(
            "Tesseract não encontrado em '%s'. "
            "Defina a variável de ambiente TESSERACT_CMD com o caminho correto.",
            _win_path,
        )
else:
    _which = shutil.which("tesseract")
    if _which is None:
        logger.warning(
            "Tesseract não encontrado no PATH. "
            "Instale-o ou defina a variável de ambiente TESSERACT_CMD."
        )

# Configurações para captura de tela
monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}

try:
    sct = mss()
except Exception:
    sct = None
    logger.error("Falha ao inicializar captura de tela (mss). A captura não funcionará.", exc_info=True)


def capturar_tela():
    """Captura a tela e retorna um frame BGR numpy array, ou None em caso de erro."""
    if sct is None:
        st.error("Captura de tela não está disponível. Verifique se o ambiente suporta mss.")
        return None
    try:
        sct_img = sct.grab(monitor)
        frame = np.array(sct_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame
    except Exception:
        logger.error("Erro ao capturar a tela.", exc_info=True)
        st.error("Erro ao capturar a tela. Verifique as configurações do monitor.")
        return None


def processar_imagem(frame):
    """Processa a imagem e identifica os números. Retorna lista de dicts ou lista vazia em caso de erro."""
    if frame is None:
        logger.warning("processar_imagem recebeu frame None, retornando lista vazia.")
        return []

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        logger.error("Falha na conversão de cor do frame.", exc_info=True)
        st.error("Erro ao processar a imagem capturada.")
        return []

    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    numeros = []
    for contorno in contornos:
        if cv2.contourArea(contorno) > 100:
            (x, y, w, h) = cv2.boundingRect(contorno)
            roi = gray[y:y + h, x:x + w]
            try:
                texto = pytesseract.image_to_string(
                    roi,
                    config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789',
                )
            except pytesseract.TesseractNotFoundError:
                logger.error("Tesseract não encontrado. Verifique a instalação.", exc_info=True)
                st.error("Tesseract OCR não encontrado. Instale-o ou configure TESSERACT_CMD.")
                return []
            except pytesseract.TesseractError:
                logger.warning("Tesseract falhou ao processar uma região da imagem.", exc_info=True)
                continue

            if texto.strip().isdigit():
                numeros.append({"x": x, "y": y, "numero": int(texto.strip())})

    numeros.sort(key=lambda n: n["x"])
    return numeros


def calcular_probabilidades(historico):
    """Calcula probabilidades. Retorna (prob_player, prob_banker) ou (None, None) se dados insuficientes."""
    if not historico:
        logger.warning("Histórico vazio, impossível calcular probabilidades.")
        return None, None
    try:
        df = pd.DataFrame(historico)
        prob_player = df["player"].value_counts(normalize=True) * 100
        prob_banker = df["banker"].value_counts(normalize=True) * 100
        return prob_player, prob_banker
    except KeyError:
        logger.error("Dados do histórico não contêm as colunas esperadas ('player', 'banker').", exc_info=True)
        return None, None


def exibir_graficos(historico):
    """Exibe gráficos de tendência. Não faz nada se dados insuficientes."""
    if not historico:
        st.info("Dados insuficientes para exibir gráficos.")
        return
    try:
        df = pd.DataFrame(historico)
        st.write("### Gráfico de Tendências")
        fig = px.line(df, x=df.index, y=["player", "banker"], title="Tendências Player vs Banker")
        st.plotly_chart(fig)
    except Exception:
        logger.error("Erro ao gerar gráficos.", exc_info=True)
        st.error("Não foi possível gerar os gráficos.")


def calcular_estatisticas(historico):
    """Calcula estatísticas. Retorna dict ou None se dados insuficientes."""
    if not historico:
        logger.warning("Histórico vazio, impossível calcular estatísticas.")
        return None
    try:
        df = pd.DataFrame(historico)
        estatisticas = {
            "Média Player": df["player"].mean(),
            "Desvio Padrão Player": df["player"].std(),
            "Média Banker": df["banker"].mean(),
            "Desvio Padrão Banker": df["banker"].std(),
        }
        return estatisticas
    except KeyError:
        logger.error("Dados do histórico não contêm as colunas esperadas ('player', 'banker').", exc_info=True)
        return None


def analisar_tela_ao_vivo():
    st.title("Análise de Bac-Bo em Tempo Real 🎲")
    st.write("Capturando e analisando a tela ao vivo...")

    if "historico" not in st.session_state:
        st.session_state.historico = []

    frame = capturar_tela()
    if frame is None:
        return

    numeros = processar_imagem(frame)

    if len(numeros) < 2:
        st.warning(
            f"Números detectados: {len(numeros)}. São necessários pelo menos 2 (Player e Banker). "
            "Verifique se a tela do jogo está visível."
        )
        return

    player = numeros[0]["numero"]
    banker = numeros[1]["numero"]
    st.write(f"Player: {player}, Banker: {banker}")

    st.session_state.historico.append({"player": player, "banker": banker})

    prob_player, prob_banker = calcular_probabilidades(st.session_state.historico)
    if prob_player is not None:
        st.write("Probabilidades Player:")
        st.write(prob_player)
        st.write("Probabilidades Banker:")
        st.write(prob_banker)
    else:
        st.warning("Não foi possível calcular probabilidades.")

    exibir_graficos(st.session_state.historico)

    estatisticas = calcular_estatisticas(st.session_state.historico)
    if estatisticas is not None:
        st.write("### Estatísticas")
        st.write(estatisticas)
    else:
        st.warning("Não foi possível calcular estatísticas.")


if __name__ == "__main__":
    analisar_tela_ao_vivo()
