# fetch/api_manager.py

import http.client
import json
from urllib.parse import urlencode
from config import API_BASE, API_KEY

API_HOST = API_BASE

HEADERS = {"X-Api-Key": API_KEY}

def fetch_endpoint(path, params=None):
    conn = http.client.HTTPSConnection(API_HOST)
    if params:
        query = urlencode(params)
        path = f"{path}?{query}"
    conn.request("GET", path, headers=HEADERS)
    res = conn.getresponse()
    data = res.read().decode("utf-8")
    conn.close()
    return json.loads(data)

def fetch_historical(stock_name, period="6m", filter="price"):
    return fetch_endpoint("/historical", {
        "stock_name": stock_name,
        "period": period,
        "filters": filter
    })
