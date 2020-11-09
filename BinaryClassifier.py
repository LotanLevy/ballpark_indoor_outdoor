
from data_tools.dataloader import Dataloader
import argparse
import os
import tensorflow as tf
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import vgg16





def get_args_parser():
    parser = argparse.ArgumentParser(description='Process constraint collector args.')
    parser.add_argument('--train_root_path',  type=str, required=True)
    parser.add_argument('--val_root_path',  type=str, required=True)

    parser.add_argument('--input_size',  type=int, default=224)
    parser.add_argument('--epochs',  type=int, default=15)

    parser.add_argument('--split_val',  type=int, default=0.2)


    parser.add_argument('--output_path', type=str, default=os.getcwd())
    return parser

def classification_architecture(input_size):
    # model = tf.keras.models.Sequential()
    # model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(input_size, input_size, 3)))
    # model.add(tf.keras.layers.Activation('relu'))
    # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(tf.keras.layers.Conv2D(32, (3, 3)))
    # model.add(tf.keras.layers.Activation('relu'))
    # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(tf.keras.layers.Conv2D(64, (3, 3)))
    # model.add(tf.keras.layers.Activation('relu'))
    # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dense(64))
    # model.add(tf.keras.layers.Activation('relu'))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Dense(1))
    # model.add(tf.keras.layers.Activation('sigmoid'))

    model = tf.keras.applications.VGG16(include_top=True, input_shape=(input_size, input_size, 3), weights='imagenet')
    prediction = tf.keras.layers.Dense(1, activation='softmax')(model.layers[-2].output)
    model = tf.keras.models.Model(inputs=model.input, outputs=prediction)

    for layer in model.layers:
        if layer.name == "fc1":
            break
        else:
            layer.trainable = False

    return model


def main():
    args = get_args_parser().parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    vgg_preprocessing = lambda input_data: vgg16.preprocess_input(np.copy(input_data.astype('float32')))

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=vgg_preprocessing, zoom_range=0.2, shear_range=0.2, horizontal_flip=True)
    train_iter = train_datagen.flow_from_directory(args.train_root_path, target_size=(args.input_size,args.input_size),
                                              batch_size=16, class_mode='binary', shuffle=True)
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=vgg_preprocessing)
    val_iter = val_datagen.flow_from_directory(args.val_root_path, target_size=(args.input_size, args.input_size),
                                              batch_size=16, class_mode='binary')



    model = classification_architecture(args.input_size)
    print(model.summary())
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    history = model.fit(
        train_iter,
        epochs=args.epochs,
        validation_data=val_iter)
    model.save_weights('binary_classification_w_after_{}_epochs.h5'.format(args.epochs))  # always save your weights after training or during training

    for item in history.history:
        plt.figure()
        plt.plot(np.arange(len( history.history[item])),  history.history[item])
        plt.title(item)
        plt.xlabel("epoch")
        plt.ylabel(item)
        plt.savefig(os.path.join(args.output_path, item))



    val_iter.reset()
    preds = model.predict(val_iter, verbose=1)
    fpr, tpr, _ = metrics.roc_curve(val_iter.classes, preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(args.output_path, "roc_curve"))
    plt.show()


if __name__ == '__main__':
    main()


