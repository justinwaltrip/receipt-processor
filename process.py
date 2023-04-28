import os
import datetime
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

# zero-shot image classification pipeline
model_name = "openai/clip-vit-large-patch14-336"
classifier = pipeline("zero-shot-image-classification", model=model_name)


def main():
    # find first empty cell in first column of google sheet
    gc = gspread.service_account("credentials.json")
    wks = gc.open_by_key(os.environ["SHEET_ID"]).sheet1
    row = len(wks.col_values(1)) + 1

    # for all pdf files in this directory
    for receipt_path in os.listdir(os.getcwd()):
        if not receipt_path.endswith(".pdf"):
            continue

        # convert receipt pdf to jpg image
        receipt_image = pdf2image.convert_from_path(receipt_path)[0]

        # get total amount from receipt
        pred = nlp(receipt_image, "What is the total amount?")
        total = float(pred[0]["answer"])

        # TODO get text from pytesseract instead and use that for zero-shot classification
        # get image class
        classes = ["Grocery", "Restaurant"]
        scores = classifier(receipt_image, candidate_labels=classes)
        image_class = scores[0]["label"]

        # # write today's date in first column
        today = datetime.date.today().strftime("%m/%d/%Y")
        wks.update_cell(row, 1, today)

        # write the total amount in second column
        wks.update_cell(row, 2, total)

        # write the image class in third column
        wks.update_cell(row, 3, image_class)

        # move to next row


if __name__ == "__main__":
    main()
