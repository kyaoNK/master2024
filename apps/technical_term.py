import openpyxl
from chardet import detect

def read_technical_term(filepath):    
    wb = openpyxl.load_workbook(filepath)

    sheet = wb['情報科全教科書用語説明付き(ver.1.2,2024-5-9)']

    technical_terms = list()

    for row in sheet.iter_rows(min_row=2):
        technical_terms.append(str(row[0].value))

    wb.close()
    
    return technical_terms

def test():
    technical_term_filepath = "/content/drive/MyDrive/Colab Notebooks/MASTER/technical_term.xlsx"
    technical_terms = read_technical_term(technical_term_filepath)
    print(technical_terms)