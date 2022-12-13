FROM python:3.9.13

RUN apt-get update && rm -rf /var/lib/apt/lists/*

WORKDIR /nlp-app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=5001" ]