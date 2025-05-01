# Bone-Fracture-Detection

## Dataset
The data set we used called MURA and included 3 different bone parts, MURA is a dataset of musculoskeletal radiographs and contains 20,335 images described below:


| **Part**     | **Normal** | **Fractured** | **Total** |
|--------------|:----------:|--------------:|----------:|
| **Elbow**    |    3160    |          2236 |      5396 |
| **Hand**     |    4330    |          1673 |      6003 |
| **Shoulder** |    4496    |          4440 |      8936 |

The data is separated into train and valid where each folder contains a folder of a patient and for each patient between 1-3 images for the same bone part

## Algorithm
Our data contains about 20,000 x-ray images, including three different types of bones - elbow, hand, and shoulder. After loading all the images into data frames and assigning a label to each image, we split our images into 72% training, 18% validation and 10% test. The algorithm starts with data augmentation and pre-processing the x-ray images, such as flip horizontal. The second step uses a ResNet50 neural network to classify the type of bone in the image. Once the bone type has been predicted, A specific model will be loaded for that bone type prediction from 3 different types that were each trained to identify a fracture in another bone type and used to detect whether the bone is fractured.
This approach utilizes the strong image classification capabilities of ResNet50 to identify the type of bone and then employs a specific model for each bone to determine if there is a fracture present. Utilizing this two-step process, the algorithm can efficiently and accurately analyze x-ray images, helping medical professionals diagnose patients quickly and accurately.
The algorithm can determine whether the prediction should be considered a positive result, indicating that a bone fracture is present, or a negative result, indicating that no bone fracture is present. The results of the bone type classification and bone fracture detection will be displayed to the user in the application, allowing for easy interpretation.
This algorithm has the potential to greatly aid medical professionals in detecting bone fractures and improving patient diagnosis and treatment. Its efficient and accurate analysis of x-ray images can speed up the diagnosis process and help patients receive appropriate care.



![img_1.png](images/Architecture.png)


## Results
### Body Part Prediction

<img src="plots/BodyPartAcc.png" width=300> <img src="plots/BodyPartLoss.png" width=300>

### Fracture Prediction
#### Elbow

<img src="plots/FractureDetection/Elbow/_Accuracy.jpeg" width=300> <img src="plots/FractureDetection/Elbow/_Loss.jpeg" width=300>

#### Hand
<img src="plots/FractureDetection/Hand/_Accuracy.jpeg" width=300> <img src="plots/FractureDetection/Hand/_Loss.jpeg" width=300>

#### Shoulder
<img src="plots/FractureDetection/Shoulder/_Accuracy.jpeg" width=300> <img src="plots/FractureDetection/Shoulder/_Loss.jpeg" width=300>


# Installations
### PyCharm IDE
### Python v3.7.x
### Install requirements.txt

* customtkinter~=5.0.3
* PyAutoGUI~=0.9.53
* PyGetWindow~=0.0.9
* Pillow~=8.4.0
* numpy~=1.19.5
* tensorflow~=2.6.2
* keras~=2.6.0
* pandas~=1.1.5
* matplotlib~=3.3.4
* scikit-learn~=0.24.2
* colorama~=0.4.5

Run mainGUI.Py

# Training for New Model

### Baseline (Original Model)
random_seed = 1

### Elbow_best
random_seed = 42
    x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(50, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)


### Hand_best
reset to baseline

AdamW(learning_rate=1e-4, weight_decay=1e-5)
callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

### Shoulder_best
reset to baseline

AdamW(learning_rate=1e-4, weight_decay=1e-5)
callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

# DEMO
### Positive Case
<img src="images/GUI/PositiveHand.png" width=400>

### Negative Case
<img src="images/GUI/NegativeHand.png" width=400>


