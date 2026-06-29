import sys
import os
from unittest.mock import patch, MagicMock

import cv2
import numpy as np
import pandas as pd
import pytest

# Add project root to path so we can import bacbobot
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock heavy/hardware-dependent modules before importing bacbobot
_mss_mock = MagicMock()
_streamlit_mock = MagicMock()
_torch_mock = MagicMock()
_torchvision_mock = MagicMock()
_pytesseract_mock = MagicMock()
_plotly_mock = MagicMock()
_matplotlib_mock = MagicMock()

with patch.dict(
    "sys.modules",
    {
        "mss": _mss_mock,
        "streamlit": _streamlit_mock,
        "torch": _torch_mock,
        "torchvision": _torchvision_mock,
        "torchvision.transforms": _torchvision_mock.transforms,
        "pytesseract": _pytesseract_mock,
        "plotly": _plotly_mock,
        "plotly.express": _plotly_mock.express,
        "matplotlib": _matplotlib_mock,
        "matplotlib.pyplot": _matplotlib_mock.pyplot,
    },
):
    import bacbobot


# ---------------------------------------------------------------------------
# Tests for calcular_probabilidades
# ---------------------------------------------------------------------------


class TestCalcularProbabilidades:
    def test_single_entry(self):
        historico = [{"player": 5, "banker": 3}]
        prob_player, prob_banker = bacbobot.calcular_probabilidades(historico)
        assert prob_player[5] == 100.0
        assert prob_banker[3] == 100.0

    def test_equal_distribution(self):
        historico = [
            {"player": 1, "banker": 2},
            {"player": 3, "banker": 4},
        ]
        prob_player, prob_banker = bacbobot.calcular_probabilidades(historico)
        assert prob_player[1] == pytest.approx(50.0)
        assert prob_player[3] == pytest.approx(50.0)
        assert prob_banker[2] == pytest.approx(50.0)
        assert prob_banker[4] == pytest.approx(50.0)

    def test_unequal_distribution(self):
        historico = [
            {"player": 7, "banker": 1},
            {"player": 7, "banker": 1},
            {"player": 7, "banker": 2},
            {"player": 3, "banker": 2},
        ]
        prob_player, prob_banker = bacbobot.calcular_probabilidades(historico)
        assert prob_player[7] == pytest.approx(75.0)
        assert prob_player[3] == pytest.approx(25.0)
        assert prob_banker[1] == pytest.approx(50.0)
        assert prob_banker[2] == pytest.approx(50.0)

    def test_all_same_values(self):
        historico = [{"player": 9, "banker": 9}] * 5
        prob_player, prob_banker = bacbobot.calcular_probabilidades(historico)
        assert prob_player[9] == pytest.approx(100.0)
        assert prob_banker[9] == pytest.approx(100.0)

    def test_many_distinct_values(self):
        historico = [{"player": i, "banker": 9 - i} for i in range(10)]
        prob_player, prob_banker = bacbobot.calcular_probabilidades(historico)
        for i in range(10):
            assert prob_player[i] == pytest.approx(10.0)
            assert prob_banker[9 - i] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Tests for calcular_estatisticas
# ---------------------------------------------------------------------------


class TestCalcularEstatisticas:
    def test_basic_statistics(self):
        historico = [
            {"player": 2, "banker": 4},
            {"player": 4, "banker": 6},
            {"player": 6, "banker": 8},
        ]
        stats = bacbobot.calcular_estatisticas(historico)
        assert stats["Média Player"] == pytest.approx(4.0)
        assert stats["Média Banker"] == pytest.approx(6.0)
        assert stats["Desvio Padrão Player"] == pytest.approx(2.0)
        assert stats["Desvio Padrão Banker"] == pytest.approx(2.0)

    def test_single_entry_stats(self):
        historico = [{"player": 5, "banker": 3}]
        stats = bacbobot.calcular_estatisticas(historico)
        assert stats["Média Player"] == pytest.approx(5.0)
        assert stats["Média Banker"] == pytest.approx(3.0)
        # std of single value is NaN
        assert pd.isna(stats["Desvio Padrão Player"])
        assert pd.isna(stats["Desvio Padrão Banker"])

    def test_identical_values(self):
        historico = [{"player": 7, "banker": 3}] * 4
        stats = bacbobot.calcular_estatisticas(historico)
        assert stats["Média Player"] == pytest.approx(7.0)
        assert stats["Média Banker"] == pytest.approx(3.0)
        assert stats["Desvio Padrão Player"] == pytest.approx(0.0)
        assert stats["Desvio Padrão Banker"] == pytest.approx(0.0)

    def test_known_std(self):
        # [1, 3] -> mean=2, sample std=sqrt(2)
        historico = [
            {"player": 1, "banker": 10},
            {"player": 3, "banker": 20},
        ]
        stats = bacbobot.calcular_estatisticas(historico)
        assert stats["Média Player"] == pytest.approx(2.0)
        assert stats["Desvio Padrão Player"] == pytest.approx(np.sqrt(2.0))
        assert stats["Média Banker"] == pytest.approx(15.0)
        assert stats["Desvio Padrão Banker"] == pytest.approx(np.sqrt(50.0))

    def test_returns_all_keys(self):
        historico = [{"player": 1, "banker": 2}]
        stats = bacbobot.calcular_estatisticas(historico)
        expected_keys = {
            "Média Player",
            "Desvio Padrão Player",
            "Média Banker",
            "Desvio Padrão Banker",
        }
        assert set(stats.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Tests for processar_imagem
# ---------------------------------------------------------------------------


class TestProcessarImagem:
    def _make_digit_image(self, digits, width=400, height=100):
        """Create a synthetic BGR image with white digits on black background."""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        x_offset = 30
        for digit in digits:
            cv2.putText(
                img,
                str(digit),
                (x_offset, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                3,
            )
            x_offset += 100
        return img

    @patch.object(bacbobot, "pytesseract")
    def test_returns_list(self, mock_tesseract):
        mock_tesseract.image_to_string.return_value = ""
        img = self._make_digit_image([])
        result = bacbobot.processar_imagem(img)
        assert isinstance(result, list)

    @patch.object(bacbobot, "pytesseract")
    def test_detects_numbers_from_ocr(self, mock_tesseract):
        mock_tesseract.image_to_string.return_value = "5\n"
        img = self._make_digit_image([5, 3])
        result = bacbobot.processar_imagem(img)
        # Each detected contour should produce a dict with x, y, numero
        for item in result:
            assert "x" in item
            assert "y" in item
            assert "numero" in item

    @patch.object(bacbobot, "pytesseract")
    def test_sorted_by_x_position(self, mock_tesseract):
        mock_tesseract.image_to_string.return_value = "1\n"
        img = self._make_digit_image([1, 2, 3])
        result = bacbobot.processar_imagem(img)
        xs = [n["x"] for n in result]
        assert xs == sorted(xs)

    @patch.object(bacbobot, "pytesseract")
    def test_filters_non_digit_text(self, mock_tesseract):
        mock_tesseract.image_to_string.return_value = "abc"
        img = self._make_digit_image([1, 2])
        result = bacbobot.processar_imagem(img)
        assert result == []

    @patch.object(bacbobot, "pytesseract")
    def test_empty_image_returns_empty(self, mock_tesseract):
        mock_tesseract.image_to_string.return_value = ""
        img = np.zeros((100, 400, 3), dtype=np.uint8)
        result = bacbobot.processar_imagem(img)
        assert result == []

    @patch.object(bacbobot, "pytesseract")
    def test_small_contours_filtered(self, mock_tesseract):
        """Contours smaller than 100px area should be ignored."""
        mock_tesseract.image_to_string.return_value = "9\n"
        # THRESH_BINARY_INV: pixels <= 200 become white (contour).
        # A bright (>200) image with a tiny dark (<=200) patch creates
        # a small contour that should be filtered out (area < 100).
        img = np.full((100, 400, 3), 255, dtype=np.uint8)
        img[50:53, 50:53] = (0, 0, 0)  # 3x3 dark patch → area ~9
        result = bacbobot.processar_imagem(img)
        assert result == []


# ---------------------------------------------------------------------------
# Tests for capturar_tela
# ---------------------------------------------------------------------------


class TestCapturarTela:
    @patch.object(bacbobot, "sct")
    def test_returns_bgr_array(self, mock_sct):
        # Simulate mss returning a BGRA screenshot
        fake_bgra = np.zeros((1080, 1920, 4), dtype=np.uint8)
        fake_bgra[:, :, 0] = 100  # B
        fake_bgra[:, :, 1] = 150  # G
        fake_bgra[:, :, 2] = 200  # R
        fake_bgra[:, :, 3] = 255  # A
        mock_sct.grab.return_value = fake_bgra

        result = bacbobot.capturar_tela()

        assert isinstance(result, np.ndarray)
        assert result.shape[2] == 3  # BGR, not BGRA
        assert result.shape[0] == 1080
        assert result.shape[1] == 1920

    @patch.object(bacbobot, "sct")
    def test_color_conversion(self, mock_sct):
        fake_bgra = np.zeros((100, 100, 4), dtype=np.uint8)
        fake_bgra[:, :, 0] = 10   # B
        fake_bgra[:, :, 1] = 20   # G
        fake_bgra[:, :, 2] = 30   # R
        fake_bgra[:, :, 3] = 255  # A
        mock_sct.grab.return_value = fake_bgra

        result = bacbobot.capturar_tela()

        # After BGRA->BGR conversion, channels should match
        assert result[0, 0, 0] == 10   # B
        assert result[0, 0, 1] == 20   # G
        assert result[0, 0, 2] == 30   # R

    @patch.object(bacbobot, "sct")
    def test_calls_grab_with_monitor(self, mock_sct):
        fake_bgra = np.zeros((1080, 1920, 4), dtype=np.uint8)
        mock_sct.grab.return_value = fake_bgra

        bacbobot.capturar_tela()

        mock_sct.grab.assert_called_once_with(bacbobot.monitor)


# ---------------------------------------------------------------------------
# Tests for exibir_graficos (smoke test with mocked Streamlit)
# ---------------------------------------------------------------------------


class TestExibirGraficos:
    def test_does_not_raise(self):
        historico = [
            {"player": 1, "banker": 2},
            {"player": 3, "banker": 4},
        ]
        # Should not raise even though streamlit is mocked
        bacbobot.exibir_graficos(historico)
