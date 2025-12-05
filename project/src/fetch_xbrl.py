import requests
import pandas as pd
import time
import os

headers = {'User-Agent': 'Bianca Poggi bpoggialways@gmail.com'}  # Replace with your info

# --- Config ---
year = 2023  # Fetch one year at a time
quarters = [1, 2, 3, 4]

xbrl_tags = [
    "Assets", "AssetsCurrent", "AssetsNoncurrent", "CashAndCashEquivalentsAtCarryingValue",
    "MarketableSecurities", "AccountsReceivableNetCurrent", "InventoryNet", "PrepaidExpenseAndOtherAssetsCurrent",
    "PropertyPlantAndEquipmentNet", "Goodwill", "IntangibleAssetsNetExcludingGoodwill",
    "DeferredTaxAssetsNetNoncurrent", "OtherAssetsNoncurrent", "Liabilities", "LiabilitiesCurrent",
    "LiabilitiesNoncurrent", "AccountsPayableCurrent", "AccruedLiabilitiesCurrent", "DeferredRevenueCurrent",
    "ShortTermBorrowings", "LongTermDebtCurrent", "LongTermDebtNoncurrent",
    "DeferredTaxLiabilitiesNoncurrent", "PensionAndOtherPostretirementDefinedBenefitPlansLiabilitiesNoncurrent",
    "StockholdersEquity", "CommonStockValue", "AdditionalPaidInCapital", "RetainedEarningsAccumulatedDeficit",
    "TreasuryStockValue", "AccumulatedOtherComprehensiveIncomeLossNetOfTax", "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax", "CostOfRevenue", "CostOfGoodsAndServicesSold",
    "GrossProfit", "OperatingExpenses", "ResearchAndDevelopmentExpense", "SellingGeneralAndAdministrativeExpense",
    "SellingAndMarketingExpense", "GeneralAndAdministrativeExpense", "DepreciationAndAmortization",
    "RestructuringCharges", "OperatingIncomeLoss", "NonoperatingIncomeExpense", "InterestExpense",
    "InterestIncomeExpenseNet", "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    "IncomeTaxExpenseBenefit", "NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic",
    "NetCashProvidedByUsedInOperatingActivities", "DepreciationDepletionAndAmortization",
    "ShareBasedCompensation", "DeferredIncomeTaxExpenseBenefit", "IncreaseDecreaseInAccountsReceivable",
    "IncreaseDecreaseInInventories", "IncreaseDecreaseInAccountsPayable", "NetCashProvidedByUsedInInvestingActivities",
    "PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireBusinessesNetOfCashAcquired",
    "PaymentsToAcquireMarketableSecurities", "ProceedsFromSaleOfPropertyPlantAndEquipment",
    "NetCashProvidedByUsedInFinancingActivities", "RepaymentsOfLongTermDebt", "ProceedsFromIssuanceOfLongTermDebt",
    "ProceedsFromIssuanceOfCommonStock", "PaymentsForRepurchaseOfCommonStock", "PaymentsOfDividends",
    "EarningsPerShareBasic", "EarningsPerShareDiluted", "WeightedAverageNumberOfSharesOutstandingBasic",
    "WeightedAverageNumberOfDilutedSharesOutstanding", "CommonStockSharesOutstanding", "CommonStockSharesIssued",
    "ComprehensiveIncomeNetOfTax", "StockIssuedDuringPeriodValueStockOptionsExercised",
    "EffectiveIncomeTaxRateContinuingOperations", "NumberOfOperatingSegments",
    "RevenueFromExternalCustomersByGeographicAreasTableTextBlock"
]

unit = 'USD'
form_types = ['3','4','5','8-K','10-K','10-Q','13D','13G','S-1']

output_file = '/workspaces/hackathon-2025-we-ll_name_this_team_later/project/data/sec_xbrl_merged.csv'

# --- Collect XBRL Data ---
xbrl_records = []
for tag in xbrl_tags:
    for q in quarters:
        url = f'https://data.sec.gov/api/xbrl/frames/us-gaap/{tag}/{unit}/CY{year}Q{q}I.json'
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                data = r.json()
                for row in data.get('data', []):
                    xbrl_records.append({
                        'cik': str(row.get('cik','')),  # Convert to string here
                        'period': f'{year}Q{q}',
                        'tag': tag,
                        'value': row.get('val', None)
                    })
                print(f"{tag} {year}Q{q} ✓ {len(data.get('data', []))} records")
            else:
                print(f"{tag} {year}Q{q} ✗ HTTP {r.status_code}")
        except Exception as e:
            print(f"{tag} {year}Q{q} ✗ {str(e)[:30]}")
        time.sleep(0.11)

df_xbrl = pd.DataFrame(xbrl_records)
if not df_xbrl.empty:
    df_xbrl = df_xbrl.pivot_table(index=['cik','period'], columns='tag', values='value').reset_index()

# --- Collect SEC Filings ---
filing_records = []
for form in form_types:
    for q in quarters:
        start_month = (q-1)*3 + 1
        end_month = q*3
        url = "https://efts.sec.gov/LATEST/search-index"
        params = {
            'category':'form-cat',
            'forms':form,
            'startdt':f'{year}-{start_month:02d}-01',
            'enddt':f'{year}-{end_month:02d}-31',
            'from':0,
            'size':1000
        }
        try:
            r = requests.get(url, params=params, headers=headers, timeout=10)
            if r.status_code == 200:
                hits = r.json().get('hits', {}).get('hits', [])
                for hit in hits:
                    src = hit.get('_source', {})
                    filing_records.append({
                        'cik': str(src.get('ciks',[''])[0]),  # Convert to string here too
                        'period': f'{year}Q{q}',
                        'form_type': form,
                        'company_name': src.get('display_names',[''])[0],
                        'filing_date': src.get('file_date',''),
                        'accession_number': src.get('accession_number','')
                    })
                print(f"{form} {year}Q{q} ✓ {len(hits)} filings")
        except Exception as e:
            print(f"{form} {year}Q{q} ✗ {str(e)[:30]}")
        time.sleep(0.11)

df_filings = pd.DataFrame(filing_records)

# --- Merge XBRL and Filings ---
df_merged = pd.merge(df_filings, df_xbrl, on=['cik','period'], how='left') if not df_xbrl.empty else df_filings

# --- Save / Append to CSV ---
os.makedirs(os.path.dirname(output_file), exist_ok=True)
df_merged.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
print(f"\nData for {year} appended to CSV: {output_file} ({len(df_merged)} rows)")
