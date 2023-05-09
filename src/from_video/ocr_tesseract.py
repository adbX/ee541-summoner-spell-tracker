import csv
import os
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image
from tqdm import tqdm


def perform_ocr(image_path, champions, spells):
    image = cv2.imread(str(image_path))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Color ranges for each text type
    colors = {
        "timestamp": [np.array([0, 0, 200]), np.array([180, 50, 255])],
        "username_champion": [np.array([100, 100, 50]), np.array([130, 255, 255])],
        "opponent_champion": [np.array([0, 100, 50]), np.array([10, 255, 255])],
        "spell": [np.array([20, 100, 150]), np.array([30, 255, 255])],
    }

    # Extract text for each color range
    extracted_texts = []
    for _, (lower, upper) in colors.items():
        mask = cv2.inRange(hsv, lower, upper)
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

        # Only find the champions and spells
        whitelist = set("".join(champions + spells))
        config = f'-c tessedit_char_whitelist="{"".join(whitelist)}" --psm 6'
        text = pytesseract.image_to_string(gray, config=config).strip()
        extracted_texts.append(text)
        # print("Extracted text:", text)

    combined_text = " ".join(extracted_texts)

    # print("Combined text:", combined_text)
    return combined_text


def process_text(text, champions, spells):
    lines = text.split("\n")
    processed_lines = []

    for line in lines:
        words = line.split()
        processed_words = []

        for word in words:
            if word in champions or word in spells:
                processed_words.append(word)

        if processed_words:
            processed_lines.append(" ".join(processed_words))

    return "\n".join(processed_lines)
