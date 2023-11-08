from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from starlette.responses import FileResponse

from src.model import spell_number

app = FastAPI()
templates = Jinja2Templates(directory='./ui/')


def save_to_text(content, filename):
    filepath = 'data/{}.txt'.format(filename)
    with open(filepath, 'w') as f:
        f.write(content)
    return filepath


@app.get('/')
def read_form():
    return templates.TemplateResponse('home.html', context={'request': request, 'result': result})

@app.get('/home')
def form_post(request: Request):
   
    return templates.TemplateResponse('home.html', context={'request': request, 'result': result})


@app.post('/seek_advice')
def form_post(request: Request, num: int = Form(...)):
    advice=request.args.get("advice")
    if advice == "Personnel Advice":
        return templates.TemplateResponse('personnel_advisor.html', context={'request': request)
    return templates.TemplateResponse('generic_advisor.html', context={'request': request)






