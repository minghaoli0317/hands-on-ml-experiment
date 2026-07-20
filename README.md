# Market Volatility Forecast

A personal machine-learning project following the end-to-end workflow
presented in Chapter 2 of Hands-On Machine Learning.

## Objective

Predict the realized volatility of a broad-market ETF over the next
20 trading days using information available at the prediction date.

## Initial scope

- Instrument: SPY
- Frequency: Daily
- Problem type: Regression
- Target: Next 20-day annualized realized volatility
- Baseline: Previous 20-day realized volatility
- Primary metric: Mean absolute error
- Validation: Chronological train, validation, and test periods

## Project stages

1. Acquire and validate the data
2. Explore the training period
3. Prepare features and targets
4. Train and compare models
5. Evaluate the selected model
6. Document conclusions and limitations

## Project structure

- `data/raw`: Unmodified source data
- `data/processed`: Prepared modeling datasets
- `notebooks`: Numbered notebooks for each project stage
- `src/market_volatility`: Reusable Python code
- `tests`: Automated tests
- `reports`: Figures and final results

## Disclaimer

This is an educational project. It does not provide investment advice
and is not intended to operate an automated trading strategy.