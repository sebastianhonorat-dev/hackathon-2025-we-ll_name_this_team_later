SECure AI – Insider Trading & SEC XBRL Anomaly Detection (Streamlit)

SECure AI is an interactive analytics tool that combines:

    Insider trading (Form 4) behavior from Finnhub
    Financial statement data (XBRL frames) + SEC filings metadata from the SEC

It builds a single, rich CSV dataset and then uses machine learning + explainability to detect and explore potential anomalies, all exposed through a Streamlit dashboard.
📌 What’s in the Main CSV File?

The core dataset is stored in:

    data/sec_xbrl_merged.csv – raw merged data (can be built by the collection script)
    data/organized_sec_xbrl_clean.csv – cleaned/processed version used for modeling

Each row in sec_xbrl_merged.csv represents:

    A company-quarter (identified by cik + period) with:

        SEC filing info (form type, company name, filing date, accession number)
        A wide set of XBRL financial features (one column per tag listed below)

🔢 Key Columns in the CSV

    cik – Central Index Key (string)
    period – Reporting period, e.g. 2023Q1
    form_type – SEC form (e.g., 10-K, 10-Q, 4, 8-K, S-1, etc.)
    company_name – Company display name
    filing_date – Filing date
    accession_number – SEC accession number

Plus one column per XBRL tag (see next section).
🧮 XBRL Features Included (Columns in the CSV)

For each cik + period, the script pulls XBRL numeric frames for the following us-gaap tags (in USD), and pivots them into columns:

    Assets
    AssetsCurrent
    AssetsNoncurrent
    CashAndCashEquivalentsAtCarryingValue
    MarketableSecurities
    AccountsReceivableNetCurrent
    InventoryNet
    PrepaidExpenseAndOtherAssetsCurrent
    PropertyPlantAndEquipmentNet
    Goodwill
    IntangibleAssetsNetExcludingGoodwill
    DeferredTaxAssetsNetNoncurrent
    OtherAssetsNoncurrent
    Liabilities
    LiabilitiesCurrent
    LiabilitiesNoncurrent
    AccountsPayableCurrent
    AccruedLiabilitiesCurrent
    DeferredRevenueCurrent
    ShortTermBorrowings
    LongTermDebtCurrent
    LongTermDebtNoncurrent
    DeferredTaxLiabilitiesNoncurrent
    PensionAndOtherPostretirementDefinedBenefitPlansLiabilitiesNoncurrent
    StockholdersEquity
    CommonStockValue
    AdditionalPaidInCapital
    RetainedEarningsAccumulatedDeficit
    TreasuryStockValue
    AccumulatedOtherComprehensiveIncomeLossNetOfTax
    Revenues
    RevenueFromContractWithCustomerExcludingAssessedTax
    SalesRevenueNet
    RevenueFromContractWithCustomerIncludingAssessedTax
    CostOfRevenue
    CostOfGoodsAndServicesSold
    GrossProfit
    OperatingExpenses
    ResearchAndDevelopmentExpense
    SellingGeneralAndAdministrativeExpense
    SellingAndMarketingExpense
    GeneralAndAdministrativeExpense
    DepreciationAndAmortization
    RestructuringCharges
    OperatingIncomeLoss
    NonoperatingIncomeExpense
    InterestExpense
    InterestIncomeExpenseNet
    IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest
    IncomeTaxExpenseBenefit
    NetIncomeLoss
    NetIncomeLossAvailableToCommonStockholdersBasic
    NetCashProvidedByUsedInOperatingActivities
    DepreciationDepletionAndAmortization
    ShareBasedCompensation
    DeferredIncomeTaxExpenseBenefit
    IncreaseDecreaseInAccountsReceivable
    IncreaseDecreaseInInventories
    IncreaseDecreaseInAccountsPayable
    NetCashProvidedByUsedInInvestingActivities
    PaymentsToAcquirePropertyPlantAndEquipment
    PaymentsToAcquireBusinessesNetOfCashAcquired
    PaymentsToAcquireMarketableSecurities
    ProceedsFromSaleOfPropertyPlantAndEquipment
    NetCashProvidedByUsedInFinancingActivities
    RepaymentsOfLongTermDebt
    ProceedsFromIssuanceOfLongTermDebt
    ProceedsFromIssuanceOfCommonStock
    PaymentsForRepurchaseOfCommonStock
    PaymentsOfDividends
    EarningsPerShareBasic
    EarningsPerShareDiluted
    WeightedAverageNumberOfSharesOutstandingBasic
    WeightedAverageNumberOfDilutedSharesOutstanding
    CommonStockSharesOutstanding
    CommonStockSharesIssued
    ComprehensiveIncomeNetOfTax
    StockIssuedDuringPeriodValueStockOptionsExercised
    EffectiveIncomeTaxRateContinuingOperations
    NumberOfOperatingSegments
    RevenueFromExternalCustomersByGeographicAreasTableTextBlock

All of these become columns in the merged CSV, so downstream modeling and visualization have access to the full financial and filings context.
🧱 Technologies Used

    Python 3.10+
    Streamlit – interactive UI
    Pandas, NumPy – data handling
    Requests – SEC API calls
    scikit-learn
        StandardScaler, PCA
        KMeans – clustering financial periods
        DBSCAN – clustering insider trades
        IsolationForest – anomaly detection
    Plotly Express – interactive 2D/3D charts
    SHAP – explain Isolation Forest anomalies
    Finnhub Python Client – Form 4 insider data
    SEC APIs
        api/xbrl/frames/us-gaap/... – XBRL numeric data
        LATEST/search-index – filings metadata

📂 Project Structure

```text
hackathon-2025-we-ll_name_this_team_later/
├── project/
│   ├── data/
│   │   └── .gitkeep                     # data folder (large CSV lives here locally)
│   └── notebooks/
│       ├── 01_xblr_exploration.ipynb   # Draft of xbrl_anomaly_pipeline.py    
│       ├── 02_form4_exploration.ipynb  # Draft of form4_insider_pipeline.py
│   └── src/
│       ├── form4_insider_pipeline.py   # Form 4 / insider trading pipeline
│       ├── xbrl_anomaly_pipeline.py    # XBRL / company facts pipeline
│       ├── fetch_xbrl.ipynb            # fetch and create xbrl.csv
├── screenshots/                        # plot screenshots
├── README.md
└── .gitignore
