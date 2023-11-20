from datetime import date
import json
from prophet_stock_trainer import StockTrainer
import os
api_key="T3G64J7EEOLUWF8K"
class PersonnelAdvisor():
  def getPortfolio(self):
        trainer = StockTrainer(api_key, "tcs")
        model, stock_history, stock_forecast = trainer.create_prophet_model(30)
        train_mean_error, test_mean_error = trainer.evaluate_prediction()
        title = 'Stock Prediction using Prophet for {} with mean error {:.2f}'.format(
            symbol, test_mean_error)
        print(title,stock_forecast[['ds', 'yhat']].to_json())
