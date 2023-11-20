from datetime import date
import json
#from .prophet_stock_trainer import StockTrainer
import os

import pandas as pd
import numpy as np

from alpha_vantage.timeseries import TimeSeries

import prophet

api_key="UR8ZCHBZ3S0A64HQ"
class PersonnelAdvisor():
  
  def getPortfolio(self,sal,age,address,gender):
        symbols=['RELIANCE.BSE','BHARTIARTL.BSE','HDFCBANK.BSE','HDFCAMC.BSE','HDFCLIFE.BSE','ASIANPAINT.BSE','INFY.BSE','ITC.BSE','DIVISLAB.BSE','TITAN.BSE','BRITANNIA.BSE']
        suggestion="Please find your allocation as below with share suggestions\n"
        if age > 20 and age <= 40:
            suggestion =suggestion+"You should invest 80% of your capital in equity, 10% in bonds,10% in gold."
        elif age > 40 and age <= 50:
            suggestion =suggestion+"You should invest 70% of your capital in equity,10% in index funds, 10% in bonds,10% in gold."   
        elif age > 40 and age <= 50:
            suggestion =suggestion+"You should invest 60% of your capital in equity,20% in index funds, 10% in bonds,10% in gold." 
        elif age > 50 and age <= 60:
            suggestion =suggestion+"You should invest 50% of your capital in equity,20% in index funds, 10% in bonds,20% in gold." 
        elif age > 50 and age <= 60:
            suggestion =suggestion+"You should invest 40% of your capital in equity,20% in index funds, 20% in bonds,30% in gold."               
        elif age > 60:
            suggestion =suggestion+"You should invest 30% of your capital in equity,10% in FD, 40% in bonds,20% in gold." 
        
        return self.predict_price(suggestion,symbols)
  
  def predict_price(self,suggestion,symbols):
    
    for sym in symbols:
            stock_file_path="../datasets/"+sym
            print(suggestion,stock_file_path)
            try:
                
                df = pd.read_csv(stock_file_path)
                print(df.head(5))
                return self.reccomend(df,None,sym,suggestion)
            except Exception as e:
                print("csv not found")
                try:
                        ts = TimeSeries(key=api_key, output_format='pandas', indexing_type='integer')
                        df, meta_data = ts.get_weekly_adjusted(symbol=sym)
                        return self.reccomend(df,stock_file_path,sym,suggestion)
                       
                except Exception as e:
                        print('Error Retrieving Data.')
                        print(e)
    return suggestion                
           
  def reccomend(self,df,stock_file_path,sym,suggestion):
        try:
            df.rename(columns={"index":"ds","4. close":"y"})
            if stock_file_path != None:
                df.to_csv(stock_file_path)
            print(df.head(5))
            fbp = prophet.Prophet(daily_seasonality = True) 
            fbp.fit(df)
            fut = fbp.make_future_dataframe(periods=365) 
            forecast = fbp.predict(fut)
            if forecast != None:
                suggestion+="\nTarget for {} on {} is {} on upside and {} on downside\n".format(sym,forecast['ds'][0],forecast['yhat'][0],forecast['yhat_lower'][0])
        except Exception as e:
             print(e)
        return suggestion

'''def getPortfolio_orig(self,sal,age,address,gender):
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
        print(symtxt,stock_forecast[['ds', 'yhat']].to_json())'''
#p=PersonnelAdvisor()
#print(p.getPortfolio(1000,42,'hyd','Male'))
