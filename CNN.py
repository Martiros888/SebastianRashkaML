import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

mnist_bldr = tfds.builder('mnist') 
mnist_bldr.download_and_prepare()
datasets = mnist_bldr.as_dataset(shuffle_files=False)
mnist_train_orig = datasets['train'] 
mnist_test_orig = datasets['test']

BUFFER_SIZE = 10000
BATCH_SIZE = 64
NUM_EPOCHS = 20
mnist_train = mnist_train_orig.map(lambda item:(tf.cast(item['image'], tf.float32) / 255.0,
                                                tf.cast(item['label'], tf.int32)))                    

mnist_test = mnist_train_orig.map(lambda item:(tf.cast(item['image'], tf.float32) / 255.0,
                                               tf.cast(item['label'], tf.int32)))

tf.random.set_seed(1)
mnist_train = mnist_train.shuffle(buffer_size=BUFFER_SIZE,reshuffle_each_iteration=False)
mnist_valid = mnist_train.take(10000).batch(BATCH_SIZE)
mnist_train = mnist_train.skip(10000).batch(BATCH_SIZE) 

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(
    filters=32, kernel_size=(5, 5),
    strides=(1, 1), padding='same',
    data_format='channels_last',
    name='conv_1', activation='relu'))

model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), name='pool_1'))
model.add(tf.keras.layers.Conv2D(
    filters=64, kernel_size=(5, 5),
    strides=(1, 1), padding='same',
    name='conv_2', activation='relu'))

model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), name='pool_2'))


model.add(tf.keras.layers.Flatten())

model.compute_output_shape(input_shape=(16, 28, 28, 1))

model.add(tf.keras.layers.Dense(
    units=1024, name='fc_1',
    activation='relu'))

model.add(tf.keras.layers.Dropout(rate=0.5))

model.add(tf.keras.layers.Dense(
    units=10, name='fc_2',
    activation='softmax'))

tf.random.set_seed(1)
model.build(input_shape=(None, 28, 28, 1))
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

history = model.fit(mnist_train, epochs=NUM_EPOCHS, 
                    validation_data=mnist_valid, shuffle=True)

hist = history.history

x_arr = np.arange(len(hist['loss'])) + 1

fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 2, 1)

ax.plot(x_arr, hist['loss'], '-o', label='poteria pri obuchenie')
ax.plot(x_arr, hist['val_loss'], '--<', label='poteria pri proverke')

ax.legend(fontsize=15)
ax.set_xlabel('epoxa', size=15)
ax.set_ylabel('poteria', size=15)
ax = fig.add_subplot(1, 2, 2)
ax.plot(x_arr, hist['accuracy'], '-o', label='pravilnost pri obuchenie')
ax.plot(x_arr, hist['val_accuracy'], '--<', label='pravlinost pri proverke')
ax.legend(fontsize=15)
ax.set_xlabel('epoxa', size=15)
ax.set_ylabel('pravlinost', size=15)
plt.show()


test_results = model.evaluate(mnist_test.batch(20))

print(test_results[1]*100)





