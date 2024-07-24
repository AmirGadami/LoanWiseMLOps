FROM python:3.8-slim-buster

RUN pip install --upgrade pip

COPY . /src

WORKDIR /src

RUN chmod +x code.py 

RUN pip install -r requirements.txt

ENTRYPOINT [ "python" ]

CMD [ "code.py" ]