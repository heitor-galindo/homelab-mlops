FROM python:3.13

RUN pip install cryptography boto3 mlflow psycopg2-binary
