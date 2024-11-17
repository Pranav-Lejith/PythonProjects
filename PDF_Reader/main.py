import pdfplumber
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt 

pdf = pdfplumber.open("tdostats.pdf")

loc_columns = ["Month_Year", "Total_Events"]


def get_total (doc):
    total_rows = []
    year_counter = 2014
    select_pages = [1,2,3,4,5,7,10,13]
    doc_pages = list(doc.pages[i] for i in select_pages)
    for page in doc_pages:
        table_first_half = page.extract_table()[-13 : -7]