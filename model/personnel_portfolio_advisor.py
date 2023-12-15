from datetime import date
import json
#from .prophet_stock_trainer import StockTrainer
import os

import pandas as pd
import numpy as np

from alpha_vantage.timeseries import TimeSeries

import prophet

api_key="RI5FRFKGBQPQGZME"
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
            stock_file_path="./datasets/"+sym
            #print(suggestion,stock_file_path)
            try:
                
                df = pd.read_csv(stock_file_path)
                suggestion= self.reccomend(df,None,sym,suggestion)
            except Exception as e:
                print("csv not found")
                try:
                        ts = TimeSeries(key=api_key, output_format='pandas', indexing_type='integer')
                        df, meta_data = ts.get_weekly_adjusted(symbol=sym)
                        suggestion= self.reccomend(df,stock_file_path,sym,suggestion)
                        print("Data downloaded")
                       
                except Exception as e:
                        print('Error Retrieving Data.')
                        print(e)
    return suggestion                
           
  def reccomend(self,df,stock_file_path,sym,suggestion):
        try:
            df=df.rename(columns={df.columns[1]: 'ds',df.columns[5]: 'y'})
            if stock_file_path != None:
                df.to_csv(stock_file_path)
            
            fbp = prophet.Prophet(daily_seasonality = True) 
            fbp.fit(df)
            fut = fbp.make_future_dataframe(periods=365) 
            forecast = fbp.predict(fut)
            latestForecast=(len(forecast.index))-1
            suggestion+="Target for {} from one year from now is {} on upside and {} on downside, as per current trend this should reach {}.".format(sym,forecast['yhat_upper'].values[latestForecast],forecast['yhat_lower'].values[latestForecast],forecast['yhat'].values[latestForecast])
        except Exception as e:
             print(e)
        return suggestion


p=PersonnelAdvisor()
print(p.getPortfolio(1000,42,'hyd','Male'))
