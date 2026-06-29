"""Visualization utilities for Streamlit dashboard."""

import plotly.express as px
import streamlit as st

from src.statistics import historico_to_dataframe


def exibir_graficos(historico):
    """Display a trend line chart of player vs banker values.

    Args:
        historico: List of dicts with 'player' and 'banker' values.
    """
    df = historico_to_dataframe(historico)
    st.write("### Gráfico de Tendências")
    fig = px.line(
        df,
        x=df.index,
        y=["player", "banker"],
        title="Tendências Player vs Banker",
    )
    st.plotly_chart(fig)
