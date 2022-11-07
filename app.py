
from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__,template_folder='template')
# MODEL_PATH='model_predict.h5'
# model=load_model(MODEL_PATH)

def model_predict(img_path):
    print(img_path)
    img=image.load_img(img_path,target_size=(224,224))
    prediction="success"
    # x=image.img_to_array(img)
    # x=x/255
    # x=np.expand_dims(x,axis=0)
    # prediction=model.predict(x)
    # prediction=np.argmax(prediction,axis=1)
    # if prediction==0:
    #     prediction="The disease in the leaf was identified as Apple Balck Rot. Captna and sulphur products are labeled for control of both scab and black rot. A scab spray program including these chemicals may help prevent the frog-eye leaf spot of black spot, as well as the infection of fruit."
    # elif prediction==1:
    #     prediction="The leaf is a fresh Apple Plant"
    # elif prediction==2:
    #     prediction="The leaf is a fresh Corn(Maize) Plant"
    # elif prediction==3:
    #     prediction="The disease in the leaf was identified as Northern Corn Leaf Blight. In addition, the rate K60 is the one effective in control of Northern Corn Leaf Blight in the field. However, there is a need for further studies in the greenhouse. Thus, the availability of fertilizer at the K60 to farmers in the endemic zones could help for sustainable management of Northern Leaf Blight in maize. "
    # elif prediction==4:
    #     prediction="The disease in the leaf was identified as Peach Bacterial Spot. Compounds available for use on peach and nectarine for bacterial spot include copper, oxytetracycline (Mycoshield and generic equivalents), and syllit+captan; however, repeated applications are typically necessary for even minimal disease control."
    # else:
    #     prediction="The leaf is a fresh Peach PLant"
    return prediction

@app.route('/')
def home():
    return render_template('signup.html')

@app.route('/dashboard',methods=['GET','POST'])
def dashboard():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f=request.files['files']
        basepath=os.path.dirname(__file__)
        file_path=os.path.join(basepath,'uploads',secure_filename(f.filename))
        f.save(file_path)

        prediction=model_predict(file_path)
        res=prediction
        return render_template('index.html',result='{}'.format(res),res=res)


    # text = request.files['file']
    # name="kavi"
    # return render_template('index.html',image='{}'.format(text),text=text)


if __name__ == '__main__':
    app.run(port=5001,debug=True) 



