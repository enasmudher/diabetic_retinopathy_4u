import base64, re
import numpy as np
import io
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify, render_template
from flask import Flask

def decode_base64(data, altchars=b'+/'):
    """Decode base64, padding being optional.

    :param data: Base64 data as an ASCII byte string
    :returns: The decoded byte string.

    """
    data = re.sub(rb'[^a-zA-Z0-9%s]+' % altchars, b'', data)  # normalize
    missing_padding = len(data) % 4
    if missing_padding:
        data += b'='* (4 - missing_padding)
    return base64.b64decode(data, altchars)

app = Flask(__name__)

def get_model():
    global model
    model = load_model('inceptionResNetv2_model_keras (2).h5')
    model._make_predict_function()
    print(" * Model Loaded!")


def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image

print(" * Loading Keras model...")
get_model()

@app.route('/')
def index():
    return render_template('predict.html')

@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image'].split(",")[0]
    encoded += "=" * ((4 - len(encoded) % 4) % 4)
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(350, 350))
    prediction = model.predict(processed_image).tolist()
    prediction = prediction[0]
    print(prediction)

    response = {
        "Normal": prediction[0],
        "Mild" : prediction[1],
        "Moderate": prediction[2],
        "Severe": prediction[3],
        "PDR": prediction[4]
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run()
