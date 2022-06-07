# Imports
import os
from turtle import down
from venv import create
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
import re

from sec_edgar_downloader import Downloader
from sec_api import ExtractorApi

def download_SEC_text(path,tickers,filing_type,after_date,before_date=None):
    dl = Downloader(f'{path}')
    for ticker in tickers:
        dl.get(filing_type, ticker, after=after_date,before=before_date)
def SEC_txt_path_finder():
    #Finds SEC Paths
    txts = []
    for root, dirs, files in os.walk(".", topdown=False):
        for name in files:
            os.path.join(root, name)
            if name[-4:]=='.txt':
                txts.append(f'{root}/{name}')
        for name in dirs:
            os.path.join(root, name)
    return txts
def path_part(path,before,after):
    beforeNum = path.find(before)+len(before)
    afterNum = path.find(after)
    return path[beforeNum:afterNum]
def sec_df_from_paths(sec_paths,filing_type):
    # Initializes a dataframe from downloaded SECs
    data = {
        'Ticker':[],
        'DocName':[],
        'DocPath':[],
        'FilingType':[],
        'DocText':[]
    }
    for sec in sec_paths:
        data['Ticker'].append(path_part(sec,'sec-edgar-filings\\',f'\\{filing_type}'))
        data['DocName'].append(path_part(sec,f'\\{filing_type}\\','/full-submission'))
        data['DocPath'].append(sec)
        data['FilingType'].append(filing_type)
        sec_txt = open(sec, "r")
        data['DocText'].append(sec_txt.read())
        sec_txt.close()
    return pd.DataFrame(data)

def DocText_extraction(text,before,after):
    """Extracts text from a starting place: before, to an end place: after"""
    return text[
        text.find(before)+len(before):
        text.find(after)
        ]
def SEC_date(DocText):
    """Returns Conformed Period of Report when given the text of an SEC report"""
    return DocText_extraction(DocText,"CONFORMED PERIOD OF REPORT:\t","\nFILED AS OF DATE:")
def SEC_CIK(DocText):
    """Returns CENTRAL INDEX KEY when given the text of an SEC report"""
    return DocText_extraction(DocText,"CENTRAL INDEX KEY:\t\t\t","\n\t\tSTANDARD INDUSTRIAL CLASSIFICATION:")
def extract_all_SEC_dates(df:pd.DataFrame):
    df['SECdates'] = df['DocText'].apply(SEC_date)
def extract_all_SEC_CIK(df:pd.DataFrame):
    df['SEC_CIK'] = df['DocText'].apply(SEC_CIK)

def OLD_url_maker(df:pd.DataFrame):
    """DOES NOT WORK
    URL maker assumed the htm file is name consistently...IT IS NOT. Need to webscrape the 
    (url_base + SEC_cik + "/" + SEC_DocName + "/") piece and find the htm file with the largest size (size is shown on the webpage)
    """
    #Example:
    # "https://www.sec.gov/Archives/edgar/data/1318605/000156459021004599/tsla-10k_20201231.htm"
    # url_base:                 https://www.sec.gov/Archives/edgar/data/
    # SEC CIK:                  1318605
    # SEC DOCName:              000156459021004599
    # Ticker:                   tsla
    # Doc Type:                 10k
    # Date (date 2020-02-31):   20201231 
    url_base = "https://www.sec.gov/Archives/edgar/data/"
    SEC_cik = df['SEC_CIK']
    SEC_DocName = df['DocName'].apply(lambda x: re.sub("-","",x))
    ticker = df['Ticker'].apply(str.lower)
    doc_type = df['FilingType'].apply(lambda x: re.sub("-","",x))
    date = df['SECdates']
    return url_base + SEC_cik + "/" + SEC_DocName + "/" + ticker + "-" + doc_type + "_" + date + '.htm'

def OLD_show_URLs(df):
    urls = OLD_url_maker(df)
    for x in urls:
        print(x)

def prepare_SEC_as_HTML(download=False):
    tickers = pd.read_csv('./Data/Cleaned_Data/Ticker_library.csv')['Ticker'].to_list()
    if download : create_SEC('./Data/Raw_Data/',tickers,"10-Q",after_date='2012-06-01')
    sec_paths = SEC_txt_path_finder()
    df_10Q = sec_df_from_paths(sec_paths,'10-Q')
    extract_all_SEC_dates(df_10Q)
    extract_all_SEC_CIK(df_10Q)
    df_10Q['urls'] = url_maker(df_10Q)
    return df_10Q

def sec_model_df():
    df = pd.read_csv('.\Data\Cleaned_Data\sec_all_data.csv',index_col=0)[[
        'ticker','periodOfReport','Business','Risk Factors',
        'Management’s Discussion and Analysis of Financial Condition and Results of Operations'
    ]]
    allText = df['Business'].fillna("")+df['Risk Factors'].fillna("")+df['Management’s Discussion and Analysis of Financial Condition and Results of Operations'].fillna("")
    sec_model = pd.concat([df,allText],axis=1).rename(columns={0:"All Text"})
    return sec_model