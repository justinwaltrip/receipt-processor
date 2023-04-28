import os
import pdf2image
from transformers import pipeline

nlp = pipeline(
    "document-question-answering",
    model="impira/layoutlm-document-qa",
)


def main():
    # for all pdf files in this directory
    for receipt_path in os.listdir(os.getcwd()):
        if not receipt_path.endswith(".pdf"):
            continue

        # convert receipt pdf to jpg image
        receipt_image = pdf2image.convert_from_path(receipt_path)[0]

        # get total amount from receipt
        pred = nlp(receipt_image, "What is the total amount?")
        total = float(pred["answer"])

        pass


if __name__ == "__main__":
    main()
