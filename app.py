from __future__ import division, print_function
# coding=utf-8
import os
import numpy as np

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from keras_preprocessing.image import load_img

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__,template_folder='template')

FRUIT_MODEL_PATH='fruit_model.h5'
model_fruit=load_model(FRUIT_MODEL_PATH)

VEG_MODEL_PATH='veg_model.h5'
model_veg=load_model(VEG_MODEL_PATH)

def model_fruitdata(img_path):
    print(img_path)
    img= load_img(img_path,target_size=(256, 256))
    i=img_to_array(img)
    im=preprocess_input(i)
    img= np.expand_dims(im, axis=0)
    prediction = np.argmax(model_fruit.predict(img))
    if prediction==0:
        prediction="The disease in the leaf was identified as Apple Balck Rot. Captna and sulphur products are labeled for control of both scab and black rot. A scab spray program including these chemicals may help prevent the frog-eye leaf spot of black spot, as well as the infection of fruit."
    elif prediction==1:
        prediction="The leaf is a fresh Apple Plant"
    elif prediction==2:
        prediction="The leaf is a fresh Corn(Maize) Plant"
    elif prediction==3:
        prediction="The disease in the leaf was identified as Northern Corn Leaf Blight. In addition, the rate K60 is the one effective in control of Northern Corn Leaf Blight in the field. However, there is a need for further studies in the greenhouse. Thus, the availability of fertilizer at the K60 to farmers in the endemic zones could help for sustainable management of Northern Leaf Blight in maize. "
    elif prediction==4:
        prediction="The disease in the leaf was identified as Peach Bacterial Spot. Compounds available for use on peach and nectarine for bacterial spot include copper, oxytetracycline (Mycoshield and generic equivalents), and syllit+captan; however, repeated applications are typically necessary for even minimal disease control."
    else:
        prediction="The leaf is a fresh Peach PLant"
    return prediction

def model_vegdata(img_path):
    print(img_path)
    img= load_img(img_path,target_size=(256, 256))
    i=img_to_array(img)
    im=preprocess_input(i)
    img= np.expand_dims(im, axis=0)
    prediction = np.argmax(model_veg.predict(img))
    if prediction==0:
        prediction="The disease in the leaf was identified as Pepper Bell Bacterial Spot.Copper sprays can be used to control bacterial leaf spot, but they are not as effective when used alone on a continuous basis. Thus, combining these sprays with a plant resistance inducer, such as Regalia or Actigard, can provide good protection from the disease."
    elif prediction==1:
        prediction="The leaf is a Healthy Pepper Bell Plant"
    elif prediction==2:
        prediction="The disease in the leaf was identified as Potato Early Blight. Mancozeb and chlorothalonil are perhaps the most frequently used protectant fungicides for early blight management but provide insufficient control under high disease pressure."
    elif prediction==3:
        prediction="The leaf is a Healthy Potato Plant."
    elif prediction==4:
        prediction="The disease in the leaf was identified as Potato Late Blight. If there is some sign of blight and the potatoes are not mature, use Dithane (mancozeb) MZ or you can also use Tattoo C or Acrobat MZ. Acrobat used later in the season reduces late blight spores."
    elif prediction==5:
        prediction="The disease in the leaf was identified as Tomato Bacterial Spot. A plant with bacterial spot cannot be cured. Remove symptomatic plants from the field or greenhouse to prevent the spread of bacteria to healthy plants. Burn, bury or hot compost the affected plants and DO NOT eat symptomatic fruit."
    elif prediction==6:
        prediction="The disease in the leaf was identified as Tomato Late Blight. Copper products can effectively control, or slow down, late blight epidemics. Copper products have no activity. Therefore, they need to be applied to all plant surfaces before infection (before symptoms are observed in the field) and frequently so new foliage is protected as plants grow."
    elif prediction==7:
        prediction="The disease in the leaf was identified as Tomato Leaf Mold. Applying fungicides when symptoms first appear can reduce the spread of the leaf mold fungus significantly. Several fungicides are labeled for leaf mold control on tomatoes and can provide good disease control if applied to all the foliage of the plant, especially the lower surfaces of the leaves."
    else:
        prediction="The disease in the leaf was identified as Tomato Septoria Leaf Spot. Most fungicides registered for use on tomatoes would effectively control Septoria leaf spot. These include maneb, mancozeb, chlorothalonil, and benomyl. Captan is not effective and zineb may be difficult to purchase."
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
        fv=request.form['select_plant']

        if fv=="fruit":
            basepath=os.path.dirname(__file__)
            file_path=os.path.join(basepath,'uploads',secure_filename(f.filename))
            f.save(file_path)
            prediction=model_fruitdata(file_path)
            res=prediction
            return render_template('index.html',result='{}'.format(res),res=res)

        else:
            basepath=os.path.dirname(__file__)
            file_path=os.path.join(basepath,'uploads',secure_filename(f.filename))
            f.save(file_path)
            prediction=model_vegdata(file_path)
            res=prediction
            return render_template('index.html',result='{}'.format(res),res=res)


if __name__ == '__main__':
    app.run(port=5001,debug=True) 




