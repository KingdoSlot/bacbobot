"""Statistical analysis utilities for game history."""

import pandas as pd


def historico_to_dataframe(historico):
    """Convert a history list to a pandas DataFrame.

    This is the single source of truth for converting the raw history
    (list of dicts with 'player' and 'banker' keys) into a DataFrame.

    Args:
        historico: List of dicts, each with 'player' and 'banker' int values.

    Returns:
        pandas.DataFrame with 'player' and 'banker' columns.
    """
    return pd.DataFrame(historico)


def calcular_probabilidades(historico):
    """Calculate value frequency distributions for player and banker.

    Args:
        historico: List of dicts with 'player' and 'banker' values.

    Returns:
        tuple: (prob_player, prob_banker) as percentage Series.
    """
    df = historico_to_dataframe(historico)
    prob_player = df["player"].value_counts(normalize=True) * 100
    prob_banker = df["banker"].value_counts(normalize=True) * 100
    return prob_player, prob_banker


def calcular_estatisticas(historico):
    """Calculate descriptive statistics for player and banker.

    Args:
        historico: List of dicts with 'player' and 'banker' values.

    Returns:
        dict with mean and standard deviation for each side.
    """
    df = historico_to_dataframe(historico)
    return {
        "Média Player": df["player"].mean(),
        "Desvio Padrão Player": df["player"].std(),
        "Média Banker": df["banker"].mean(),
        "Desvio Padrão Banker": df["banker"].std(),
    }
