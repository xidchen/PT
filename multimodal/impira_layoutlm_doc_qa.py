# !apt install tesseract-ocr
# !pip install pytesseract

import transformers


dqa = transformers.pipeline(
    task="document-question-answering",
    model="impira/layoutlm-document-qa",
)

dqa(
    "https://templates.invoicehome.com/invoice-template-us-neat-750px.png",
    "Who signs this invoice?"
)
