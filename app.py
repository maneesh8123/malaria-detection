import streamlit as st
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
import os

model= load_model('model_detectt.h5')

st.title("Malaria Detection")






def main(img, model):
    img = image.load_img(img, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)

    ## Scaling
    x = x / 255
    x = np.expand_dims(x, axis=0)


    x = preprocess_input(x)

    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds == 0:
        preds = "Infected With Pneumonia"
    else:
        preds = " not Infected With Pneumonia"

    return preds



uploaded_file = st.file_uploader('image', type=['png', 'jpeg'])



image_path=os.path.join(r'Dataset')


preds=main(image_path,model)
preds



if __name__=='__main__':
    main()







