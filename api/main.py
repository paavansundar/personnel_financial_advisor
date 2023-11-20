import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from starlette.responses import FileResponse
from fastapi.staticfiles import StaticFiles


from typing import Annotated

from model import generic_advice,share_specific_advice,personnel_portfolio_advisor

app = FastAPI()
app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent.parent.absolute() / "static"),
    name="static",
)
templates = Jinja2Templates(directory='./templates')

@app.get('/')
def read_form(request: Request):
    return templates.TemplateResponse('home.html',context={'request': request})

@app.get('/home')
def form_home(request: Request):
   
    return templates.TemplateResponse('home.html', context={'request': request})


@app.post('/seek_advice')
def form_post(request: Request,advice: Annotated[str, Form()]):
    #advice=request.args.get("advice")
    if advice == "Personnel Advice":
        return templates.TemplateResponse('personnel_advisor.html', context={'request': request})
    elif advice == "Stock Advice":
        return templates.TemplateResponse('stock_specific_query.html', context={'request': request})
    
    return templates.TemplateResponse('generic_advisor.html', context={'request': request})

@app.post('/personnel_advice')
def form_post(request: Request,sal: Annotated[int, Form()],
          age: Annotated[int, Form()],
          gender: Annotated[str, Form()],
          address: Annotated[str, Form()],
          ):
              
    perPortfolioObj=personnel_portfolio_advisor.PersonnelAdvisor()
    portfolio=perPortfolioObj.getPortfolio(sal,age,address,gender)
    return templates.TemplateResponse('personnel_advisor.html', context={'request': request,'response':portfolio})

@app.post('/stock_specific')
def form_post(request: Request,sal: Annotated[int, Form()],
          age: Annotated[int, Form()],
          gender: Annotated[str, Form()],
          address: Annotated[str, Form()],
          stock: Annotated[str, Form()]):
    print("age",age,gender,address,stock)
    stockObj=share_specific_advice.StockSpecific()
    res=stockObj.getStockAdvice(sal,age,address,gender,stock)
    return templates.TemplateResponse('stock_specific_query.html', context={'request': request,'response':res})

@app.post('/general_advice')
def form_post(request: Request,prompt: Annotated[str, Form()]):
    print(prompt)
    genericAdviceObj=generic_advice.GenericAdvice()
    answer=genericAdviceObj.chat(prompt)
    return templates.TemplateResponse('generic_advisor.html', context={'request': request,'answer':answer})


