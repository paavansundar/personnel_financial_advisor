from datetime import date
import json
from .prophet_stock_trainer import StockTrainer
import os
api_key="T3G64J7EEOLUWF8K"
class PersonnelAdvisor():
  def getPortfolio(self,sal,age,address,gender):
        symbols=['RELIANCE.BSE','BHARTIARTL.BSE','HDFCBANK.BSE','HDFCAMC.BSE','HDFCLIFE.BSE','ASIANPAINT.BSE','INFY.BSE','ITC.BSE','DIVISLAB.BSE','TITAN.BSE','BRITANNIA.BSE']
        suggestion="Please find your allocation as below with share suggestion as below<br/>"
        if age > 30 and age < 40:
            suggestion =suggestion+"You should invest 80% of your capital in equity, 10% in bonds,10% in gold."
        elif age > 40 and age < 50:
            suggestion =suggestion+"You should invest 70% of your capital in equity,10% in index funds, 10% in bonds,10% in gold."   
        elif age > 40 and age < 50:
            suggestion =suggestion+"You should invest 60% of your capital in equity,20% in index funds, 10% in bonds,10% in gold." 
        elif age > 50 and age < 60:
            suggestion =suggestion+"You should invest 50% of your capital in equity,20% in index funds, 10% in bonds,20% in gold." 
        elif age > 50 and age < 60:
            suggestion =suggestion+"You should invest 40% of your capital in equity,20% in index funds, 20% in bonds,30% in gold."               
        elif age > 60:
            suggestion =suggestion+"You should invest 30% of your capital in equity,10% in FD, 40% in bonds,20% in gold." 
        return suggestion
    
  def getPortfolio_orig(self,sal,age,address,gender):
        symbols=['RELIANCE.BSE','BHARTIARTL.BSE','HDFCBANK.BSE','HDFCAMC.BSE','HDFCLIFE.BSE','ASIANPAINT.BSE','INFY.BSE','ITC.BSE','DIVISLAB.BSE','TITAN.BSE','BRITANNIA.BSE']
        suggestion="Please find your allocation as below with share suggestion as below<br/>"
        if age > 30 and age < 40:
            suggestion =suggestion+"You should invest 80% of your capital in equity, 10% in bonds,10% in gold."
        elif age > 40 and age < 50:
            suggestion =suggestion+"You should invest 70% of your capital in equity,10% in index funds, 10% in bonds,10% in gold."   
        elif age > 40 and age < 50:
            suggestion =suggestion+"You should invest 60% of your capital in equity,20% in index funds, 10% in bonds,10% in gold." 
        elif age > 50 and age < 60:
            suggestion =suggestion+"You should invest 50% of your capital in equity,20% in index funds, 10% in bonds,20% in gold." 
        elif age > 50 and age < 60:
            suggestion =suggestion+"You should invest 40% of your capital in equity,20% in index funds, 20% in bonds,30% in gold."               
        elif age > 60:
            suggestion =suggestion+"You should invest 30% of your capital in equity,10% in FD, 40% in bonds,20% in gold." 
        symtxt=""
        for symbol in symbols: 
          trainer = StockTrainer(api_key, symbol)
          model, stock_history, stock_forecast = trainer.create_prophet_model(30)
          train_mean_error, test_mean_error = trainer.evaluate_prediction()
          print(symbol,stock_forecast[['ds', 'yhat']].to_json())
          symtxt+=symbol+"<br/>"
        symtxt = '<u>Suggested stocks</u><br/> {}'.format(
            symtxt)
        print(symtxt,stock_forecast[['ds', 'yhat']].to_json())
p=PersonnelAdvisor()
p.getPortfolio(1000,42,'hyd','Male')
