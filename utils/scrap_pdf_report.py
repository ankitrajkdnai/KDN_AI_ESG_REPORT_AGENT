import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

# Load the CSV file containing company names & symbols
df = pd.read_csv('SP50List.csv')
company_name = df['Name']
company_symbol = df['Symbol']

# Base URL
BASE_URL = "https://www.responsibilityreports.com"

# Set the storage directory
REPORTS_DIR = "./gcs-bucket/reports/"

# Ensure the directory exists
os.makedirs(REPORTS_DIR, exist_ok=True)

def get_esg_report(symbol):
    try:
        # Step 1: Perform the search request
        payload = {'search': symbol}
        search_response = requests.post(f"{BASE_URL}/Companies", data=payload)

        if search_response.status_code != 200:
            print(f"Error: Failed to fetch search results for {symbol}. Status code: {search_response.status_code}")
            return
        
        # Step 2: Parse search results page
        search_soup = BeautifulSoup(search_response.text, 'html.parser')
        company_link = search_soup.select_one('.companyName a')
        
        if not company_link:
            print(f"Error: No ESG report found for {symbol}")
            return
        
        company_url = BASE_URL + company_link['href']
        
        # Step 3: Fetch the company page
        company_response = requests.get(company_url)
        if company_response.status_code != 200:
            print(f"Error: Failed to load company page for {symbol}")
            return

        company_soup = BeautifulSoup(company_response.text, 'html.parser')
        report_link_element = company_soup.select_one('.view_btn a')

        if not report_link_element:
            print(f"Error: No report link found for {symbol}")
            return
        
        report_url = BASE_URL + report_link_element['href']
        
        # Step 4: Download the ESG report
        report_response = requests.get(report_url)
        if report_response.status_code != 200:
            print(f"Error: Failed to download report for {symbol}")
            return
        
        # Save the report in the specified directory
        year = '2025'
        file_name = f"{year}_{symbol}.pdf"
        file_path = os.path.join(REPORTS_DIR, file_name)

        with open(file_path, 'wb') as f:
            f.write(report_response.content)
        
        print(f"✅ ESG report saved at {file_path}")

    except Exception as e:
        print(f"⚠️ Error processing {symbol}: {e}")

# Process each company in the CSV
for symbol in company_symbol[:40]:  # Limit to first 40 companies for testing
    get_esg_report(symbol)
