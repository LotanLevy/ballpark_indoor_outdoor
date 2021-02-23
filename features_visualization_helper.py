
from tensorflow.keras import preprocessing
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import cv2
import matplotlib.pyplot as plt

def gradCAM_by_feature(path, preprocess_func, model, feature_idx, conv_layer='block5_conv3',  intensity=0.3, res=250):
    img = preprocessing.image.load_img(path, target_size=(224, 224))

    x = preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_func(x)

    preds = model.predict(x)

    with tf.GradientTape() as tape:
        last_conv_layer = model.get_layer(conv_layer)
        iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
        model_out, last_conv_layer = iterate(x)
        class_out = model_out[:, feature_idx]
        grads = tape.gradient(class_out, last_conv_layer)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = heatmap.reshape((14, 14))

    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))


    heatmap = cv2.cvtColor(cv2.applyColorMap((heatmap*255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)

    img =  cv2.addWeighted(heatmap , intensity, img, 1-intensity, 0)/255

    return img

