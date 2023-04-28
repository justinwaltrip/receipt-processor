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

# # zero-shot image classification pipeline
# model_name = "openai/clip-vit-large-patch14-336"
# classifier = pipeline("zero-shot-image-classification", model=model_name)
classifier = None


def main():
    # find first empty cell in first column of google sheet
    gc = gspread.service_account()
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
        total = float(pred["answer"])

        # get image class
        classes = ["Grocery", "Restaurant"]
        scores = classifier(receipt_image, classes)
        image_class = scores[0]["label"]

        # # write today's date in first column
        today = datetime.date.today().strftime("%m/%d/%Y")
        # service.spreadsheets().values().update(
        #     spreadsheetId=os.environ["SHEET_ID"],
        #     range=f"A{row}",
        #     valueInputOption="RAW",
        #     body={"values": [[today]]},
        # ).execute()

        # # write the total amount in second column
        # service.spreadsheets().values().update(
        #     spreadsheetId=os.environ["SHEET_ID"],
        #     range=f"B{row}",
        #     valueInputOption="RAW",
        #     body={"values": [[total]]},
        # ).execute()

        # # write the image class in third column
        # service.spreadsheets().values().update(
        #     spreadsheetId=os.environ["SHEET_ID"],
        #     range=f"C{row}",
        #     valueInputOption="RAW",
        #     body={"values": [[image_class]]},
        # ).execute()

        # combine above api calls into one batch request
        batch_update_values_request_body = {
            "value_input_option": "RAW",
            "data": [
                {
                    "range": f"A{row}",
                    "values": [[today]],
                },
                {
                    "range": f"B{row}",
                    "values": [[total]],
                },
                {
                    "range": f"C{row}",
                    "values": [[image_class]],
                },
            ],
        }
        service.spreadsheets().values().batchUpdate(
            spreadsheetId=os.environ["SHEET_ID"],
            body=batch_update_values_request_body,
        ).execute()


if __name__ == "__main__":
    main()
