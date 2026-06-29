import logging
import os

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import pytesseract
import streamlit as st
from mss import mss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tesseract path from environment variable (falls back to system default)
tesseract_cmd = os.environ.get("TESSERACT_CMD")
if tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

# Monitor config from environment or defaults
MONITOR = {
    "top": int(os.environ.get("MONITOR_TOP", "0")),
    "left": int(os.environ.get("MONITOR_LEFT", "0")),
    "width": int(os.environ.get("MONITOR_WIDTH", "1920")),
    "height": int(os.environ.get("MONITOR_HEIGHT", "1080")),
}


def capturar_tela():
    """Capture the screen region defined by MONITOR."""
    try:
        sct = mss()
        sct_img = sct.grab(MONITOR)
        frame = np.array(sct_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame
    except Exception:
        logger.exception("Failed to capture screen")
        return None


def processar_imagem(frame):
    """Process the captured frame and extract numbers via OCR."""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contornos, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        numeros = []
        for contorno in contornos:
            if cv2.contourArea(contorno) > 100:
                (x, y, w, h) = cv2.boundingRect(contorno)
                roi = gray[y : y + h, x : x + w]
                texto = pytesseract.image_to_string(
                    roi,
                    config="--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789",
                )
                stripped = texto.strip()
                if stripped.isdigit():
                    numeros.append({"x": x, "y": y, "numero": int(stripped)})

        numeros.sort(key=lambda n: n["x"])
        return numeros
    except Exception:
        logger.exception("Failed to process image")
        return []


def calcular_probabilidades(historico):
    """Calculate historical probabilities for player and banker outcomes."""
    df = pd.DataFrame(historico)
    prob_player = df["player"].value_counts(normalize=True) * 100
    prob_banker = df["banker"].value_counts(normalize=True) * 100
    return prob_player, prob_banker


def exibir_graficos(historico):
    """Display trend chart for player vs banker."""
    df = pd.DataFrame(historico)
    st.write("### Trend Chart")
    fig = px.line(
        df,
        x=df.index,
        y=["player", "banker"],
        title="Player vs Banker Trends",
    )
    st.plotly_chart(fig)


def calcular_estatisticas(historico):
    """Calculate descriptive statistics for the game history."""
    df = pd.DataFrame(historico)
    return {
        "Player Mean": df["player"].mean(),
        "Player Std Dev": df["player"].std(),
        "Banker Mean": df["banker"].mean(),
        "Banker Std Dev": df["banker"].std(),
    }


def analisar_tela_ao_vivo():
    """Main Streamlit entrypoint for live Bac-Bo analysis."""
    st.title("Bac-Bo Live Analysis")
    st.write("Capturing and analyzing screen in real time...")

    if "historico" not in st.session_state:
        st.session_state.historico = []

    frame = capturar_tela()
    if frame is None:
        st.error("Screen capture failed. Check logs for details.")
        return

    numeros = processar_imagem(frame)

    if len(numeros) >= 2:
        player = numeros[0]["numero"]
        banker = numeros[1]["numero"]
        st.write(f"Player: {player}, Banker: {banker}")

        st.session_state.historico.append({"player": player, "banker": banker})

        prob_player, prob_banker = calcular_probabilidades(st.session_state.historico)
        st.write("Player Probabilities:")
        st.write(prob_player)
        st.write("Banker Probabilities:")
        st.write(prob_banker)

        exibir_graficos(st.session_state.historico)

        estatisticas = calcular_estatisticas(st.session_state.historico)
        st.write("### Statistics")
        st.write(estatisticas)
    else:
        st.warning("Could not detect at least 2 numbers on screen.")


if __name__ == "__main__":
    analisar_tela_ao_vivo()
