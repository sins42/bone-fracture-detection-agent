import numpy as np
import pandas as pd
import os.path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import wandb

import random

from tensorflow_addons.optimizers import AdamW

import wandb
from wandb.sklearn import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from tensorflow.keras.metrics import AUC


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

class CustomWandbCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            wandb.log(logs, step=epoch)

class LogLearningRate(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            lr = lr(epoch)
        wandb.log({"learning_rate": float(tf.keras.backend.get_value(lr))}, step=epoch)

# load images to build and train the model
#                       ....                                     /    img1.jpg
#             test      Hand            patient0000   positive  --   img2.png
#           /                /                         \    .....
#   Dataset   -         Elbow  ------   patient0001
#           \ train               \         /                           img1.png
#                       Shoulder        patient0002     negative --      img2.jpg
#                       ....                   \
#

def load_path(path, part):
    """
    load X-ray dataset
    """
    dataset = []
    for folder in os.listdir(path):
        folder = path + '/' + str(folder)
        if os.path.isdir(folder):
            for body in os.listdir(folder):
                if body == part:
                    body_part = body
                    path_p = folder + '/' + str(body)
                    for id_p in os.listdir(path_p):
                        patient_id = id_p
                        path_id = path_p + '/' + str(id_p)
                        for lab in os.listdir(path_id):
                            if lab.split('_')[-1] == 'positive':
                                label = 'fractured'
                            elif lab.split('_')[-1] == 'negative':
                                label = 'normal'
                            path_l = path_id + '/' + str(lab)
                            for img in os.listdir(path_l):
                                img_path = path_l + '/' + str(img)
                                dataset.append(
                                    {
                                        'body_part': body_part,
                                        'patient_id': patient_id,
                                        'label': label,
                                        'image_path': img_path
                                    }
                                )
    return dataset


# this function get part and know what kind of part to train, save model and save plots
def trainPart(part):
    # categories = ['fractured', 'normal']
    # Initialize wandb
    wandb.init(
        project="FractureDetection",
        name=f"ResNet50-{part}",
        config={
            "model": "ResNet50",
            "part": part,
            "epochs": 25,
            "batch_size": 64,
            "optimizer": "Adam",
            "learning_rate": 0.0001
        },
        reinit=True
    )
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    image_dir = THIS_FOLDER + '/Dataset/'
    data = load_path(image_dir, part)
    labels = []
    filepaths = []

    # add labels for dataframe for each category 0-fractured, 1- normal
    for row in data:
        labels.append(row['label'])
        filepaths.append(row['image_path'])

    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    images = pd.concat([filepaths, labels], axis=1)

    # split all dataset 10% test, 90% train (after that the 90% train will split to 20% validation and 80% train
    train_df, test_df = train_test_split(images, train_size=0.9, shuffle=True, random_state=1)

    # each generator to process and convert the filepaths into image arrays,
    # and the labels into one-hot encoded labels.
    # The resulting generators can then be used to train and evaluate a deep learning model.

    # now we have 10% test, 72% training and 18% validation
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,
                                                                      preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
                                                                      validation_split=0.2)

    # use ResNet50 architecture
    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input)

    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=64,
        shuffle=True,
        seed=42,
        subset='training'
    )

    val_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=64,
        shuffle=True,
        seed=42,
        subset='validation'
    )

    test_images = test_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False, 
        seed = SEED
    )

    # we use rgb 3 channels and 224x224 pixels images, use feature extracting , and average pooling
    pretrained_model = tf.keras.applications.resnet50.ResNet50(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg')

    # for faster performance
    pretrained_model.trainable = False

    for layer in pretrained_model.layers[-20:]:
        layer.trainable = True

    inputs = pretrained_model.input
    x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
    x = tf.keras.layers.Dense(50, activation='relu')(x)

    # outputs Dense '2' because of 2 classes, fratured and normal
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    optimizer = Adam(learning_rate=0.0001)
    # print(model.summary())
    print("-------Training " + part + "-------")

    # Adam optimizer with low learning rate for better accuracy
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', AUC(name='auc')])

    # early stop when our model is over fit or vanishing gradient, with restore best values
    callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(train_images, validation_data=val_images, epochs=25, callbacks=[callbacks, CustomWandbCallback(), LogLearningRate()])

    # Log confusion matrix
    y_pred = model.predict(val_images)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = val_images.classes
    class_names = list(val_images.class_indices.keys())

    wandb.sklearn.plot_confusion_matrix(y_true, y_pred_classes, labels=class_names, title=f"{part} Confusion Matrix")


    # save model to this path
    results = model.evaluate(test_images, verbose=0)
    wandb.log({"Test Loss": results[0], "Test Accuracy": results[1]})
    print(part + " Results:")
    print(results)
    print(f"Test Accuracy: {np.round(results[1] * 100, 2)}%")

    model_path = THIS_FOLDER + f"/new_weights/ResNet50_{part}_frac.h5"
    model.save(model_path)
    wandb.save(model_path)

# run the function and create model for each parts in the array
categories_parts = ["Elbow", "Hand", "Shoulder"]
for category in categories_parts:
    trainPart(category)
