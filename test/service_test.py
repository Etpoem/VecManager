# -*- coding: UTF-8 -*-
# -------------------------------------------------------------
# @Author   :   panjf
# @Time     :   2021/2/19 16:14
# @Desc     :   
# -------------------------------------------------------------
import datetime
import time
import json
import numpy as np
import requests

BASE_URL = "http://192.168.96.138:17100"


def db_create():
    url = f'{BASE_URL}/faiss/db/create'
    data = {
        'db_name': 'temp1',
        'max_size': 1000,
        'info': '测试',
        'is_multiple_gpus': True
    }
    response = requests.post(url=url, json=data)
    print(response.json())


def db_info():
    url = f'{BASE_URL}/faiss/db/info'
    data = {
        'db_name': ''
    }
    response = requests.post(url=url, json=data)
    print(response.json())


def db_remove():
    url = f'{BASE_URL}/faiss/db/remove'
    data = {
        'db_name': 'temp1'
    }
    response = requests.post(url=url, json=data)
    print(response.json())


def db_update():
    url = f'{BASE_URL}/faiss/db/update'
    data = {
        'db_name': 'temp1',
        'max_size': 2000
    }
    response = requests.post(url=url, json=data)
    print(type(response.json()))
    print(response.json())


def feature_insert():
    url = f'{BASE_URL}/faiss/feature/insert'
    data = {
        'db_name': 'temp1',
        'feature_id': 1,
        'feature': json.dumps(np.random.random([512]).astype('float32').tolist()),
        'faceCoord': [1, 2, 3, 4],
        'info': '',
        'create_datetime': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    response = requests.post(url=url, json=data)
    print(response.json())


def feature_delete():
    url = f'{BASE_URL}/faiss/feature/delete'
    data = {
        'db_name': 'temp1',
        'feature_id': 1
    }
    response = requests.post(url=url, json=data)
    print(response.json())


def search_feature():
    url = f'{BASE_URL}/faiss/feature/search'
    data = {
        'db_name': "temp_1",
        'feature': json.dumps(np.random.random([512]).astype('float32').tolist())
    }
    response = requests.post(url=url, json=data)
    print(response.json())


def faiss_state():
    url = f'{BASE_URL}/faiss/state'
    response = requests.get(url=url)
    print(response.json())


if __name__ == "__main__":
    # db_create()
    db_info()
    # db_remove()
    # db_update()
    # feature_insert()
    # feature_delete()
    # search_feature()
    # faiss_state()
