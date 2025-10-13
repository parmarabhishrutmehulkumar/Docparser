import os
import pdfplumber
from pathlib import Path
import shutil
import sys
import numpy as np
# Try to import pytesseract and EasyOCR (optional)
PYTESSERACT_AVAILABLE = False
EASYOCR_AVAILABLE = False
try:
    import pytesseract
    from PIL import Image
    PYTESSERACT_AVAILABLE = True
except Exception:
    PYTESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except Exception:
    EASYOCR_AVAILABLE = False

# If pytesseract imported, try to locate tesseract binary
TESSERACT_CMD = None
if PYTESSERACT_AVAILABLE:
    # prefer system PATH
    found = shutil.which("tesseract")
    if found:
        TESSERACT_CMD = found
    else:
        # common Windows install path
        possible = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if Path(possible).exists():
            TESSERACT_CMD = possible
        else:
            possible_x86 = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
            if Path(possible_x86).exists():
                TESSERACT_CMD = possible_x86

    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    else:
        # disable pytesseract usage if binary not found
        PYTESSERACT_AVAILABLE = False

PDF_PATH = Path(r"c:\Users\ABHISHRUT\Desktop\NidhiSK Resume.pdf")
OUT_PATH = Path(r"c:\Users\ABHISHRUT\Desktop\docparser\pdftest_output.txt")

if not PDF_PATH.exists():
    raise SystemExit(f"File not found: {PDF_PATH}")

print(f"Opening: {PDF_PATH}  (size={PDF_PATH.stat().st_size} bytes)")

full_text = []

# initialize EasyOCR reader once (if available)
reader = None
if EASYOCR_AVAILABLE:
    try:
        # CPU mode; set gpu=True if you have CUDA and proper torch installed
        reader = easyocr.Reader(['en'], gpu=False)
    except Exception as e:
        print(f"EasyOCR initialization error: {e}")
        reader = None
        EASYOCR_AVAILABLE = False

with pdfplumber.open(str(PDF_PATH)) as pdf:
    num_pages = len(pdf.pages)
    print(f"Pages: {num_pages}")

    for i, page in enumerate(pdf.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            full_text.append(text)
            print(f"Page {i}: extracted text length={len(text)}")
            continue

        print(f"Page {i}: no embedded text found")
        ocr_text = None

        # Try pytesseract OCR
        if PYTESSERACT_AVAILABLE:
            try:
                img = page.to_image(resolution=600).original
                ocr_text = pytesseract.image_to_string(img)
                if ocr_text and ocr_text.strip():
                    full_text.append(ocr_text)
                    print(f"Page {i}: pytesseract OCR extracted length={len(ocr_text)}")
                    continue
                else:
                    print(f"Page {i}: pytesseract returned no text")
            except Exception as e:
                print(f"Page {i}: pytesseract error: {e}")

        # Try EasyOCR OCR as fallback (pass numpy array)
        if EASYOCR_AVAILABLE and reader is not None:
            try:
                pil_img = page.to_image(resolution=600).original.convert("RGB")
                img_np = np.array(pil_img)
                results = reader.readtext(img_np)
                if results and len(results) > 0:
                    ocr_text = "\n".join([r[1] for r in results])
                    full_text.append(ocr_text)
                    print(f"Page {i}: EasyOCR extracted length={len(ocr_text)}")
                    continue
                else:
                    print(f"Page {i}: EasyOCR returned no text")
            except Exception as e:
                print(f"Page {i}: EasyOCR error: {e}")

        print(f"Page {i}: OCR not available or returned no text")

combined = "\n\n".join(full_text).strip()

if not combined:
    print("No text extracted from PDF (embedded text not found and OCR not available or returned empty).")
    # actionable instructions
    if not PYTESSERACT_AVAILABLE and not EASYOCR_AVAILABLE:
        print("\nOptions to enable OCR:")
        print("A) Install Tesseract (recommended):")
        print("   - Download Windows installer (e.g. UB-Mannheim builds) from:")
        print("     https://github.com/UB-Mannheim/tesseract/wiki")
        print("   - Or official releases: https://github.com/tesseract-ocr/tesseract/releases")
        print("   - After install, verify in a new terminal: tesseract --version")
        print("   - If tesseract.exe is not on PATH, set the path in this script or add to PATH.")
        print("\nB) Use EasyOCR (pure-Python fallback):")
        print("   - pip install easyocr")
        print("   - pip will also require a torch backend; see EasyOCR docs for installation notes.")
        print("\nAfter installing, re-run this script.")
    elif not PYTESSERACT_AVAILABLE:
        print("\nPytesseract is installed but Tesseract binary not found. Install Tesseract and ensure it's on PATH.")
    elif not EASYOCR_AVAILABLE:
        print("\nEasyOCR not installed; you can install it with: pip install easyocr")
else:
    print(f"Total extracted characters: {len(combined)}")
    OUT_PATH.write_text(combined, encoding="utf-8")
    print(f"Saved extracted text to: {OUT_PATH}")
    print("\n--- Preview (first 1000 chars) ---\n")
    print(combined[:1000000])
    print("\n--- End Preview ---\n")