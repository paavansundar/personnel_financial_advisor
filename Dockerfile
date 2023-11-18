FROM python:3.10

RUN mkdir -p personnel_financial_advisor
RUN echo "$PWD"
ADD . /personnel_financial_advisor

WORKDIR /personnel_financial_advisor
RUN echo "ls"

RUN pip install --no-cache-dir -r ./requirements/requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "./api/main.py"]
