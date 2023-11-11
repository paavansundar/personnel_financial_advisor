from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from starlette.responses import FileResponse

from model import GenericAdvice,StockSpecific



app = FastAPI()
templates = Jinja2Templates(directory='./ui')


def save_to_text(content, filename):
    filepath = 'data/{}.txt'.format(filename)
    with open(filepath, 'w') as f:
        f.write(content)
    return filepath


@app.get('/')
def read_form():
    return templates.TemplateResponse('home.html')

@app.get('/home')
def form_home(request: Request):
   
    return templates.TemplateResponse('home.html', context={'request': request})


@app.post('/seek_advice')
def form_post(request: Request):
    advice=request.args.get("advice")
    if advice == "Personnel Advice":
        return templates.TemplateResponse('personnel_advisor.html', context={'request': request})
    elif advice == "Stock Advice":
        return templates.TemplateResponse('stock_specific_query.html', context={'request': request})
    
    return templates.TemplateResponse('generic_advisor.html', context={'request': request})

@app.post('/personnel_advice')
def form_post(request: Request):
    advice=request.args.get("advice")
    return templates.TemplateResponse('personnel_advisor.html', context={'request': request})

@app.post('/stock_specific')
def form_post(request: Request):
    sal=request.args.get("sal")
    age=request.args.get("age")
    address=request.args.get("address")
    gender=request.args.get("gender")
    stock=request.args.get("stock") 
    stockObj=StockSpecific()
    return stockObj.getStockAdvice(sal,age,address,gender,stock)

@app.post('/generic_advice')
def form_post(request: Request):
    prompt=request.args.get("prompt")
    genericAdviceObj=GenericAdvice()
    answer=genericAdviceObj.chat(prompt)
    return templates.TemplateResponse('generic_advisor.html', context={'request': request,'answer':answer})
