# base image

FROM python:3.12.6

# workdir

WORKDIR /app

# copy

COPY . /app

# run

pip install -r requirements.txt

# port

EXPOSE 8000

# command

CMD ["uvicorn", "main:app", "--host"]