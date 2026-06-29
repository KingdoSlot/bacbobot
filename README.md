# bacbobot

Real-time Bac-Bo game analysis bot using screen capture, OCR, and statistical modeling.

## Project Structure

```
src/
├── __init__.py          # Package init
├── app.py               # Main Streamlit application
├── capture.py           # Screen capture utilities
├── image_processing.py  # OCR and image analysis
├── statistics.py        # Probability and descriptive stats
└── visualization.py     # Plotly/Streamlit charts
```

## Setup

```bash
pip install -r requirements.txt
```

Ensure [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) is installed and update the path in `src/image_processing.py` if needed.

## Run

```bash
streamlit run src/app.py
```
