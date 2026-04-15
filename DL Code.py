
#Dataset Link :  https://www.kaggle.com/ciplab/real-and-fake-face-detection



import os

BASE_DIR = "real_and_fake_face"
train_real = os.path.join(BASE_DIR, "training_real")
train_fake = os.path.join(BASE_DIR, "training_fake")

print(len(os.listdir(train_real)))
print(len(os.listdir(train_fake)))

import random
import matplotlib.pyplot as plt
from PIL import Image

def show_images(folder, title):
    images = os.listdir(folder)
    plt.figure(figsize=(10,4))
    for i in range(5):
        img_path = os.path.join(folder, random.choice(images))
        img = Image.open(img_path)
        plt.subplot(1,5,i+1)
        plt.imshow(img)
        plt.axis("off")
    plt.suptitle(title)
    plt.show()

show_images(train_real, "REAL IMAGES")
show_images(train_fake, "FAKE IMAGES")

import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_dataset = tf.keras.utils.image_dataset_from_directory(
    BASE_DIR,
    labels='inferred',
    label_mode='binary',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True
)

for images, labels in train_dataset.take(1):
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Labels:", labels.numpy())

train_ds = tf.keras.utils.image_dataset_from_directory(
    BASE_DIR,
    labels='inferred',
    label_mode='binary',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    validation_split=0.2,
    subset="training",
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    BASE_DIR,
    labels='inferred',
    label_mode='binary',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=42
)

from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(224, 224, 3)),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(224, 224, 3)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

history_tl = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

base_model.trainable = True


for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

history_ft = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3
)

base_model.trainable = False
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

import numpy as np

y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy().astype(int).flatten())
    y_pred.extend((preds > 0.5).astype(int).flatten())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("Samples evaluated:", len(y_true))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Fake (0)", "Real (1)"]
)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (Validation)")
plt.show()

from sklearn.metrics import classification_report

print(classification_report(
    y_true, y_pred,
    target_names=["Fake", "Real"]
))

base_model = model.get_layer("mobilenetv2_1.00_224")

last_conv_layer = None

for layer in base_model.layers[::-1]:
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer = layer.name
        break

print("Last convolution layer:", last_conv_layer)



def grad_cam(model, base_model, img_array, layer_name):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [base_model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, prediction = grad_model(img_array)
        loss = prediction[:, 0]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

for images, labels in val_ds.take(1):
    img = images[0]
    label = labels[0].numpy()
    break

img_array = tf.expand_dims(img, axis=0)





_ = model(img_array)


heatmap = grad_cam(
    model,
    base_model,
    img_array,
    last_conv_layer
)

from tensorflow.keras.applications.efficientnet import preprocess_input

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

base_model_eff = EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)


base_model_eff.trainable = False


efficientnet_model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(224, 224, 3)),
    base_model_eff,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

efficientnet_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

efficientnet_model.summary()


history_eff = efficientnet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt


y_true_eff = []
y_pred_eff = []

for images, labels in val_ds:
    preds = efficientnet_model.predict(images, verbose=0)
    y_true_eff.extend(labels.numpy().astype(int).flatten())
    y_pred_eff.extend((preds > 0.5).astype(int).flatten())

y_true_eff = np.array(y_true_eff)
y_pred_eff = np.array(y_pred_eff)

print("Samples evaluated (EfficientNet):", len(y_true_eff))


cm_eff = confusion_matrix(y_true_eff, y_pred_eff)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_eff,
    display_labels=["Fake (0)", "Real (1)"]
)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (EfficientNet)")
plt.show()

# t
print("EfficientNet Classification Report:\n")
print(classification_report(
    y_true_eff,
    y_pred_eff,
    target_names=["Fake", "Real"]
))

base_model_eff.trainable = True


for layer in base_model_eff.layers[:-30]:
    layer.trainable = False


efficientnet_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_eff_ft = efficientnet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

from sklearn.utils.class_weight import compute_class_weight
import numpy as np


labels = []

for _, y in train_ds:
    labels.extend(y.numpy().astype(int).flatten())

labels = np.array(labels)


class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)

class_weights = {i: class_weights[i] for i in range(len(class_weights))}

print("Class Weights:", class_weights)

history_eff_final = efficientnet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    class_weight=class_weights
)

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models
import tensorflow as tf

base_model_eff = EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

base_model_eff.trainable = False


efficientnet_model = models.Sequential([
    layers.Lambda(preprocess_input, input_shape=(224, 224, 3)),
    base_model_eff,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

efficientnet_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_eff = efficientnet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=8
)

base_model_eff.trainable = True

for layer in base_model_eff.layers[:-20]:
    layer.trainable = False

efficientnet_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_ft = efficientnet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

y_true_eff = []
y_pred_eff = []

for images, labels in val_ds:
    preds = efficientnet_model.predict(images, verbose=0)
    y_true_eff.extend(labels.numpy().astype(int).flatten())
    y_pred_eff.extend((preds > 0.5).astype(int).flatten())

y_true_eff = np.array(y_true_eff)
y_pred_eff = np.array(y_pred_eff)

print("Samples evaluated (EfficientNet):", len(y_true_eff))


cm_eff = confusion_matrix(y_true_eff, y_pred_eff)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_eff,
    display_labels=["Fake (0)", "Real (1)"]
)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (EfficientNet)")
plt.show()

print("EfficientNet Classification Report:\n")
print(classification_report(
    y_true_eff,
    y_pred_eff,
    target_names=["Fake", "Real"]
))

efficientnet_model.save("deepfake_model_eff.keras")

print("Model saved successfully!")