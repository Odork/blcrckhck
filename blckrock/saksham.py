import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings
#from google.colab import drive

warnings.filterwarnings('ignore')

def read_synthetic_csv(filename):
    df = pd.read_csv(filename)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    else:
        print("No 'Date' column found. Generating a time series index.")  # Debug print
        df.index = pd.date_range(start='2010-01-01', periods=len(df), freq='M')
    return df

def fit_arima_models(df):
    models = {}
    predictions = {}
    
    for column in df.columns:
        print(f"Fitting ARIMA model for column: {column}")  # Debug print
        model = ARIMA(df[column], order=(1,1,1))
        model_fit = model.fit()
        models[column] = model_fit
        predictions[column] = model_fit.forecast(steps=1)[0]
    
    return predictions

def create_user_profile(amount, years, roi, risk, monthly_income, savings, monthly_investment):
    return pd.DataFrame({
        'Amount': [amount],
        'Years': [years],
        'ROI': [roi],
        'Risk': [risk],
        'MonthlyIncome': [monthly_income],
        'Savings': [savings],
        'MonthlyInvestment': [monthly_investment]
    })

def wealth_management_model(amount, years, roi, risk, monthly_income, savings, monthly_investment, csv_file, emergency_fund_percentage=0.2):
    # Read synthetic data from CSV
    df = read_synthetic_csv(csv_file)
    print("DataFrame head:\n", df.head())  #e Debug print
    
    # Fit ARIMA models and get predictions
    predictions = fit_arima_models(df)
    
    # Create user profile
    user_profile = create_user_profile(amount, years, roi, risk, monthly_income, savings, monthly_investment)
    
    # Calculate emergency fund
    emergency_fund = emergency_fund_percentage * monthly_income * 6  # Emergency fund for 6 months
    
    # Adjust the total investment amount after setting aside emergency fund
    adjusted_amount = amount - emergency_fund
    
    # Normalize predictions to get allocation percentages
    total_predicted_value = sum(predictions.values())
    allocation_percentages = {k: v / total_predicted_value for k, v in predictions.items()}
    
    portfolio = {
        'Stocks': allocation_percentages['Stocks'] * adjusted_amount,
        'Bonds': allocation_percentages['Bonds'] * adjusted_amount,
        'MutualFunds': allocation_percentages['MutualFunds'] * adjusted_amount,
        'RealEstate': allocation_percentages['RealEstate'] * adjusted_amount,
        'Commodities': allocation_percentages['Commodities'] * adjusted_amount,
        'ESG': allocation_percentages['ESG'] * adjusted_amount,
        'EmergencyFund': emergency_fund
    }
    
    return portfolio

# Mount Google Drive
#drive.mount('/content/drive')

# Example usage with CSV file from Google Drive
csv_file = 'synthetic_portfolio_data.csv'
portfolio = wealth_management_model(200000, 8, 0.08, 0.05, 6000, 10000, 2000, csv_file)
print(portfolio)
