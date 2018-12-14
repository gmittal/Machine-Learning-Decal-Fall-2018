from tensorflow.keras.models import load_model
from util import *
model = load_model('save/model.h5')
def predict(sent):
    return model.predict(embedding(sent), steps=1)
