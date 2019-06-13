# get python image
FROM  python:alpine2.7

# workdir 
COPY  . /app
WORKDIR /app

# install necessary software
Run   apt-get update
Run   apt-get install python-dev python-pip
RUN   pip install -r requirements.txt

# run python command
CMD  ["python", "RunConfigLabyrinth.py"]
