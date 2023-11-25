from pathlib import Path
env_path = Path('.') / '.env'
import json
import os
import requests
import pandas as pd
import pickle

#api_key = os.environ.get("KZSCEANOC0AZFNX5")
api_key=""
class StockSpecific:
    def get_tolerance(self,user):
        #tol_port,tol_stock="",""
     #try:
        data = pd.read_csv("../datasets/calculate_tolerance.csv", delimiter=",", index_col=False)
        #Extract user information to find tolerance
        sal = user['sal']
        age = int(user['age'])
        res = user['res']
        gender = user['gender']
        #get salary category
        if (sal<=250000):
            sal_cat = 'L'
        elif (sal>250000) & (sal<=1000000):
            sal_cat = 'LM'
        elif (sal>1000000) & (sal<=3000000):
            sal_cat = 'UM'
        elif (sal>3000000):
            sal_cat = 'H'
    #Narrow down according to above data and find appropriate tolerance
        temp = data.loc[(data['Age_Min'] <= age) & (data['Age_Max'] >= age)]
        temp1 = temp.loc[(temp['Gender'] == gender) & (temp['Residency'] == res) & (temp['Sal_Cat'] == sal_cat)]
        #Extract and return tolerance values
        tol_port = temp1['Tol_P']
        tol_stock = temp1['Tol_S']
     #except Exception as e:
       # print(e)
        return tol_port, tol_stock

    def stock_info(self,ticker, tol):

        reco=""
        tol=2
        tol1 = float((100+tol)/100) # Format the tolerance value to float
        tol = str(tol)
        # obtain the data from ALPHAVANTAGE for that ticker
        request_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={api_key}"
        response = requests.get(request_url)
        parsed_response = json.loads(response.text)
        print("Response is ",parsed_response)
        last_refreshed = parsed_response["Meta Data"]["3. Last Refreshed"]
        tsd = parsed_response["Time Series (Daily)"]
        dates = list(tsd.keys())
        latest_day = dates[0]
        latest_close = tsd[latest_day]["4. close"]
        latest_close1 = "INR {0:,.2f}".format(float(latest_close)*60.0)
        high_prices = []
        low_prices = []
        for date in dates:
            high_price = tsd[date]["2. high"]
            low_price = tsd[date]["3. low"]
            high_prices.append(float(high_price))
            low_prices.append(float(low_price))
        recent_high = max(high_prices)
        recent_high1 = "INR {0:,.2f}".format(float(recent_high)*60.0)
        recent_low = min(low_prices)
        recent_low1 = "INR {0:,.2f}".format(float(recent_low)*60.0)
        print(f"Latest available data: {last_refreshed}")
        print(f"Latest closing price: {latest_close1} ")
        print(f"Recent avergae closing high price: {recent_high1}")
        print(f"Recent average closing low price: {recent_low1}")
        print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-")
        threshold = tol1*(float(recent_low)*60.0)
        if (float(latest_close)*60.0) < threshold:
            reco="Recommendation: BUY!,Reason: The latest closing price is not greater than"+tol+"% of the recent low,indicating potential growth. Go for it!"
        else:
            reco="Recommendation: DON'T BUY!,Reason: The latest closing price is greater than"+tol+"% of the recent low, indicating potential decline. Umm, maybe next time?"
        return reco

    def digi_crypto_info(self,frm, to):
        print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-")
        # obtain the data from ALPHAVANTAGE
        request_url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={frm}&to_currency={to}&apikey={api_key}"
        response = requests.get(request_url)
        parsed_response = json.loads(response.text)
        #Extract all necessary columns
        parsed_response = parsed_response["Realtime Currency Exchange Rate"]
        last_refreshed = parsed_response["6. Last Refreshed"]
        frm_code = parsed_response["1. From_Currency Code"]
        to_code = parsed_response["3. To_Currency Code"]
        frm_curr = parsed_response["2. From_Currency Name"]
        to_curr = parsed_response["4. To_Currency Name"]
        time_zone = parsed_response["7. Time Zone"]
        rate = parsed_response["5. Exchange Rate"]
        #Print out results
        print(f"Latest available data: {last_refreshed}({time_zone})")
        print(f"{frm_curr}({frm_code}) ---> {to_curr}({to_code}): {rate}")
        print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-")

    def getStockAdvice(self,sal,age,res,gender,ticker): 
        user=dict()
        user['sal']=sal
        user['age']=age
        user['res']=res
        user['gender']=gender
        #tol=self.get_tolerance(user) #Need Recheck
        return self.stock_info(ticker, 2)
  
