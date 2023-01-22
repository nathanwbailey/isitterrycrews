from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import sys
sys.path.append('/Users/nathanbailey/Documents/terrycrews')
import eval_net

app = Flask(__name__)


@app.route("/send_image", methods=['POST'])
def send_image():
    print(request.headers)
    image = request.files['image']
    image.save('/Users/nathanbailey/Documents/terrycrews/react_terrycrews/terrycrews/image.png')
    pred, per_1, per_2 = eval_net.main('/Users/nathanbailey/Documents/terrycrews/react_terrycrews/terrycrews/image.png')
    per_1 = int(per_1*100)
    per_2 = int(per_2*100)
    data = {'output': pred, 'per_1': per_1, 'per_2': per_2}
    print(data)
    data = jsonify(data)
    return data, 200

