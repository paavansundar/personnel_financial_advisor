FROM python:3.10

RUN mkdir -p personnel_financial_advisor
RUN pwd
ADD ./personnel_financial_advisor personnel_financial_advisor

WORKDIR /personnel_financial_advisor

RUN pip install --no-cache-dir -r ./requirements/requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "./api/main.py"]
