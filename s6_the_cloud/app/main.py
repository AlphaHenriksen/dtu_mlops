from fastapi import FastAPI
from http import HTTPStatus
import re
from pydantic import BaseModel
from fastapi import UploadFile, File
from typing import Optional
# import cv2
from fastapi.responses import FileResponse


app = FastAPI()

class Email(BaseModel):
    email: str
    domain: str | None = None


@app.get("/")
def root():
    """ Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response

from enum import Enum
class ItemEnum(Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

@app.get("/query_items")
def read_item(item_id: int):
    return {"item_id": item_id}

database = {'username': [ ], 'password': [ ]}

@app.get("/text_model/")
def contains_email(data: Email):
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    response = data
    # response = {
    #     "input": data,
    #     "message": HTTPStatus.OK.phrase,
    #     "status-code": HTTPStatus.OK,
    #     "is_email": re.fullmatch(regex, data) is not None
    # }
    return response


@app.post("/login/")
def login(username: str, password: str):
    username_db = database['username']
    password_db = database['password']
    if username not in username_db and password not in password_db:
        with open('database.csv', "a") as file:
            file.write(f"{username}, {password} \n")
        username_db.append(username)
        password_db.append(password)
        return "login saved"
    else:
        return "Logged in"


@app.post("/cv_model/")
async def cv_model(data: UploadFile = File(...), h: int = None, w: int = None):
    with open('image.jpg', 'wb') as image:
        content = await data.read()
        image.write(content)
        image.close()

    # if h and w:
    #     image = cv2.imread("image.jpg")
    #     res = cv2.resize(image, (h, w))
    #     cv2.imwrite('image_resize.jpg', res)
    #     FileResponse('image_resize.jpg')
    FileResponse('image.jpg')

    response = {
        "input": data,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response
