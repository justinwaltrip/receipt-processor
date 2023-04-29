import os
from dateutil.parser import parse
import pytesseract
from typing import Optional
import pdf2image
from transformers import pipeline
from dotenv import load_dotenv
import gspread


load_dotenv("config.env")  # take environment variables from .env.

# document question answering model
nlp = pipeline(
    "document-question-answering",
    model="impira/layoutlm-document-qa",
)

# zero-shot text classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]


def apply_tesseract(
    image: "Image.Image",
    lang: Optional[str] = "eng",
    tesseract_config: Optional[str] = "",
):
    """Applies Tesseract OCR on a document image, and returns recognized words + normalized bounding boxes."""
    # apply OCR
    data = pytesseract.image_to_data(
        image, lang=lang, output_type="dict", config=tesseract_config
    )
    words, left, top, width, height = (
        data["text"],
        data["left"],
        data["top"],
        data["width"],
        data["height"],
    )

    # filter empty words and corresponding coordinates
    irrelevant_indices = [idx for idx, word in enumerate(words) if not word.strip()]
    words = [word for idx, word in enumerate(words) if idx not in irrelevant_indices]
    left = [coord for idx, coord in enumerate(left) if idx not in irrelevant_indices]
    top = [coord for idx, coord in enumerate(top) if idx not in irrelevant_indices]
    width = [coord for idx, coord in enumerate(width) if idx not in irrelevant_indices]
    height = [
        coord for idx, coord in enumerate(height) if idx not in irrelevant_indices
    ]

    # turn coordinates into (left, top, left+width, top+height) format
    actual_boxes = []
    for x, y, w, h in zip(left, top, width, height):
        actual_box = [x, y, x + w, y + h]
        actual_boxes.append(actual_box)

    image_width, image_height = image.size

    # finally, normalize the bounding boxes
    normalized_boxes = []
    for box in actual_boxes:
        normalized_boxes.append(normalize_box(box, image_width, image_height))

    if len(words) != len(normalized_boxes):
        raise ValueError("Not as many words as there are bounding boxes")

    return words, normalized_boxes


def main():
    # find first empty cell in first column of google sheet
    gc = gspread.service_account("credentials.json")
    wks = gc.open_by_key(os.environ["SHEET_ID"]).sheet1
    row = len(wks.col_values(1)) + 1

    # store values
    values = []

    # for all pdf files in this directory
    for receipt_path in os.listdir(os.getcwd()):
        if not receipt_path.endswith(".pdf"):
            continue

        # convert receipt pdf to jpg image
        receipt_image = pdf2image.convert_from_path(receipt_path)[0]
        receipt_image = receipt_image.convert("RGB")

        # perform ocr
        words, boxes = apply_tesseract(receipt_image)

        # get total amount from receipt
        word_boxes = list(zip(words, boxes))
        pred = nlp(receipt_image, "What is the total amount?", word_boxes=word_boxes)
        total = float(pred[0]["answer"])

        # get date from receipt
        pred = nlp(receipt_image, "What is the date?", word_boxes=word_boxes)
        date = parse(pred[0]["answer"])

        # get receipt class
        text = " ".join(words)
        classes = ["Grocery", "Restaurant"]
        scores = classifier(text, classes)
        image_class = scores["labels"][0]

        # store values
        values.append((date, total, image_class))

    # sort values by date
    values = sorted(values, key=lambda x: x[0])

    # write values to sheet
    for date, total, image_class in values:
        wks.update(f"A{row}:C{row}", [[date.strftime("%-m/%d/%Y"), total, image_class]])
        row += 1

    # delete all pdf files
    for receipt_path in os.listdir(os.getcwd()):
        if receipt_path.endswith(".pdf"):
            os.remove(receipt_path)

    print("Done!")


if __name__ == "__main__":
    main()
