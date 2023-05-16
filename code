import tensorflow as tf
from tensorflow.keras import models,layers
import matplotlib.pyplot as plt

IMAGE_SIZE=256
BATCH_SIZE=32

from google.colab import drive
drive.mount('/content/gdrive')

!ln -s /content/gdrive/My\ Drive/ /mydrive
!ls

DIRECTORY = "/mydrive/PlantVillage"

dataset=tf.keras.preprocessing.image_dataset_from_directory(
    DIRECTORY,
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

class_names=dataset.class_names
class_names

len(dataset)

for image_batch,label_batch in dataset.take(1):
    print(image_batch[0].shape)
    plt.imshow(image_batch[0].numpy().astype('uint8'))
    plt.axis('off')
    plt.title(class_names[label_batch[0]])
    
    len(dataset)
    
    train_size=0.8
len(dataset)*train_size

test_ds=dataset.skip(57)
len(test_ds)

val_size=0.1
len(dataset)*val_size

val_ds=test_ds.take(7)
len(val_ds)

test_ds=test_ds.skip(7)
len(test_ds)

def get_dataset_partitions_tf(ds,train_split=0.8,val_split=0.1,test_split=0.1,shuffle=True,shuffle_size=1000):
    ds_size=len(ds)
    if shuffle:
        ds=ds.shuffle(shuffle_size,seed=12)
    train_size=int(train_split*ds_size)
    val_size=int(val_split*ds_size)
    
    train_ds=dataset.take(train_size)
    val_ds=dataset.skip(train_size).take(val_size)
    test_ds=dataset.skip(train_size).skip(val_size)
    
    return train_ds,val_ds,test_ds
    
    train_ds,val_ds,test_ds=get_dataset_partitions_tf(dataset)
    
    len(train_ds)
    
    len(val_ds)
    
    len(test_ds)
    
    train_ds=train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds=val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds=test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

resize_and_rescale=tf.keras.Sequential([layers.experimental.preprocessing.Resizing(IMAGE_SIZE,IMAGE_SIZE),
                                        layers.experimental.preprocessing.Rescaling(1.0/255)])
                                        
                                        tf.get_logger().setLevel('ERROR')
                                        
                                        data_augmentation=tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),
])
channels=3

input_shape=(BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,channels)
n_classes=3
model= models.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(32,(3,3),activation='relu',input_shape=input_shape),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(n_classes,activation='softmax')
])
model.build(input_shape=input_shape)

model.summary()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

history=model.fit(
    train_ds,
    epochs=50,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_data=val_ds
    
)

scores=model.evaluate(test_ds)
scores

history

history.params

history.history.keys()
import numpy as np

for images_batch ,labels_batch in test_ds.take(1):
    first_image=image_batch[0].numpy().astype('uint8')
    first_label=labels_batch[0].numpy()
    plt.imshow(first_image)
    print(class_names[first_label])
    batch_prediction=model.predict(images_batch)
    print(class_names[np.argmax(batch_prediction[0])])
    
    def predict(model,img):
    image_array=tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    image_array=tf.expand_dims(image_array,0)
    
    predictions=model.predict(image_array)
    
    predicted_class=class_names[np.argmax(predictions[0])]
    confidence=round(100*(np.max(predictions[0])),2)
    return predicted_class,confidence
    
    BATCH_SIZE=9
plt.figure(figsize=(15,15))
for images,labels in test_ds.take(1):
    for i in range(BATCH_SIZE):
        ax=plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        predicted_class,confidence=predict(model,images[i].numpy())
        actual_class=class_names[labels[i]]
        plt.title(f"Actual:{actual_class}.\npredicted:{predicted_class}.\nconfidance:{confidence}%")
        plt.axis('off')
        
        history.history.keys()
        
        acc =history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(range(len(acc)), acc)
plt.plot(range(len(val_acc)), val_acc[:len(acc)])

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(range(len(loss)), loss)
plt.plot(range(len(val_loss)), val_loss[:len(loss)])

model.save("cofee_disease.model",save_format="h5")
