"""Main Streamlit application for Bac-Bo real-time analysis."""

import streamlit as st

from src.capture import capturar_tela
from src.image_processing import processar_imagem
from src.statistics import calcular_estatisticas, calcular_probabilidades
from src.visualization import exibir_graficos


def analisar_tela_ao_vivo():
    """Main analysis loop: capture screen, detect numbers, show stats."""
    st.title("Análise de Bac-Bo em Tempo Real 🎲")
    st.write("Capturando e analisando a tela ao vivo...")

    if "historico" not in st.session_state:
        st.session_state.historico = []

    frame = capturar_tela()
    numeros = processar_imagem(frame)

    if len(numeros) >= 2:
        player = numeros[0]["numero"]
        banker = numeros[1]["numero"]
        st.write(f"Player: {player}, Banker: {banker}")

        st.session_state.historico.append({"player": player, "banker": banker})

        prob_player, prob_banker = calcular_probabilidades(
            st.session_state.historico
        )
        st.write("Probabilidades Player:")
        st.write(prob_player)
        st.write("Probabilidades Banker:")
        st.write(prob_banker)

        exibir_graficos(st.session_state.historico)

        estatisticas = calcular_estatisticas(st.session_state.historico)
        st.write("### Estatísticas")
        st.write(estatisticas)


if __name__ == "__main__":
    analisar_tela_ao_vivo()
