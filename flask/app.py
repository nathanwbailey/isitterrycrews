from flask import Flask, session
import flask
import requests
from flask import request
import os
import base64
import json
import google.oauth2.credentials
import google_auth_oauthlib.flow
import sys
from flask_session import Session
from flask_cors import CORS
from flask import jsonify
sys.path.append('/Users/nathanbailey/Documents/terrycrews/')
import eval_net
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

app = Flask(__name__)
app.secret_key = 'hi.'
CORS(app, supports_credentials=True)

google_credentials = None
spotify_token = None

app.config['SESSION_TYPE'] = 'filesystem'
@app.route("/callnetwork")
def callnetwork():
    data = request.get_json(force=True)
    url=data['url']
    eval_net.main()


