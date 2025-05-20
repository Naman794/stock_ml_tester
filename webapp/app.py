# webapp/app.py

from flask import Flask
from webapp.routes import main

def create_app():
    app = Flask(__name__)
    app.register_blueprint(main)
    return app
