"""Image processing and OCR utilities."""

import cv2
import pytesseract

# Tesseract configuration (adjust path for your OS)
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Arquivos de Programas\Tesseract-OCR\tesseract.exe"
)

TESSERACT_CONFIG = "--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789"
MIN_CONTOUR_AREA = 100


def processar_imagem(frame):
    """Process a frame to extract numbers via OCR.

    Converts to grayscale, applies thresholding, finds contours,
    and uses Tesseract to read digits from each region of interest.

    Args:
        frame: BGR image (numpy.ndarray).

    Returns:
        list[dict]: Detected numbers sorted left-to-right,
                    each with keys 'x', 'y', 'numero'.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contornos, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    numeros = []
    for contorno in contornos:
        if cv2.contourArea(contorno) < MIN_CONTOUR_AREA:
            continue
        (x, y, w, h) = cv2.boundingRect(contorno)
        roi = gray[y : y + h, x : x + w]
        texto = pytesseract.image_to_string(roi, config=TESSERACT_CONFIG)
        if texto.strip().isdigit():
            numeros.append({"x": x, "y": y, "numero": int(texto.strip())})

    numeros.sort(key=lambda n: n["x"])
    return numeros
