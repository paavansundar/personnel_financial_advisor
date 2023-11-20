FROM python:3.10

RUN mkdir -p personnel_financial_advisor

ADD . /personnel_financial_advisor

WORKDIR /personnel_financial_advisor

RUN pip install --no-cache-dir -r ./requirements/requirements.txt

CMD ["uvicorn", "app.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]

