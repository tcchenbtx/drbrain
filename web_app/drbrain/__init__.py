# import os
# from flask import Flask, request, redirect, url_for
# from werkzeug.utils import secure_filename
# from flaskexample import views

from flask import Flask
app = Flask(__name__)
from drbrain import views


