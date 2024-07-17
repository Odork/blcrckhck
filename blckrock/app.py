from flask import Flask, request, jsonify, render_template  
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import logging
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

logging.basicConfig(level=logging.DEBUG)

# Function to read synthetic data from CSV
def read_synthetic_csv(filename):
    df = pd.read_csv(filename)
    if 'Date' not in df.columns:
        logging.debug("No 'Date' column found. Generating a time series index.")
        df.index = pd.date_range(start='1/1/2010', periods=len(df), freq='M')
    else:
        df.set_index('Date', inplace=True)
    logging.debug("DataFrame head:\n%s", df.head())
    return df

# Function to fit ARIMA models
def fit_arima_models(df):
    models = {}
    predictions = {}

    for column in df.columns:
        logging.debug("Fitting ARIMA model for column: %s", column)
        model = ARIMA(df[column], order=(1,1,1))
        model_fit = model.fit()
        models[column] = model_fit
        predictions[column] = model_fit.forecast(steps=1)[0]

    return predictions

# Function to create user profile
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

# Main function to compute wealth management model
def wealth_management_model(amount, years, roi, risk, monthly_income, savings, monthly_investment, csv_file, emergency_fund_percentage=0.2):
    df = read_synthetic_csv(csv_file)
    predictions = fit_arima_models(df)
    user_profile = create_user_profile(amount, years, roi, risk, monthly_income, savings, monthly_investment)
    emergency_fund = emergency_fund_percentage * monthly_income * 6
    adjusted_amount = amount - emergency_fund
    total_predicted_value = sum(predictions.values())
    allocation_percentages = {k: v / total_predicted_value for k, v in predictions.items()}

    portfolio = {
        'Stocks': allocation_percentages.get('Stocks', 0) * adjusted_amount,
        'Bonds': allocation_percentages.get('Bonds', 0) * adjusted_amount,
        'MutualFunds': allocation_percentages.get('MutualFunds', 0) * adjusted_amount,
        'RealEstate': allocation_percentages.get('RealEstate', 0) * adjusted_amount,
        'Commodities': allocation_percentages.get('Commodities', 0) * adjusted_amount,
        'ESG': allocation_percentages.get('ESG', 0) * adjusted_amount,
        'EmergencyFund': emergency_fund
    }

    return portfolio

@app.route('/')
def index():
    return render_template('index.html')  # Ensure 'index.html' is in the 'templates' folder

# API endpoint to handle POST requests from frontend
@app.route('/api/portfolio', methods=['POST'])
def portfolio():
    try:
        logging.debug("Received request with data: %s", request.json)
        data = request.json
        amount = float(data['amount'])
        years = int(data['years'])
        roi = float(data['roi'])
        risk = float(data['risk'])
        monthly_income = float(data['monthly_income'])
        savings = float(data['savings'])
        monthly_investment = float(data['monthly_investment'])
        csv_file = 'synthetic_portfolio_data.csv'
        result = wealth_management_model(amount, years, roi, risk, monthly_income, savings, monthly_investment, csv_file)
        logging.debug("Generated portfolio: %s", result)
        return jsonify(result), 200
    except Exception as e:
        logging.error("Error processing request: %s", str(e))
        return jsonify({'error': 'An error occurred. Please check your inputs and try again.'}), 400

if __name__ == '__main__':
    app.run(debug=True)