FROM python:3.10
WORKDIR /s6_the_cloud
COPY ./requirements.txt /s6_the_cloud/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /s6_the_cloud/requirements.txt
COPY ./app /s6_the_cloud/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]

# docker build -f docker.dockerfile . -t fastapi
# docker run --name mycontainer -p 80:80 fastapi