# %% [markdown]

"""
Homework:

The folder '~//data//homework' contains a folder 'Data', containing hand-digits of letters a-z stored in .txt.

Try to establish a network to classify the digits.

`dataLoader.py` offers APIs for loading data.
"""
# %%
import dataLoader as dl

features,labels=dl.readData(r'../data/homework/Data')
labels_ = [ord(x.lower())-97 for x in labels]
# %%
import matplotlib.pyplot as plt
plt.plot(features[5,0:30],features[5,30:])
plt.suptitle="Real: "+''.format(labels[5])
plt.show()
# %%
# feature engineering (if necessary)
import string
class_names = list(string.ascii_lowercase)
class_names
# %%
# train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=.2, random_state=1)
# %%
# build the network
import tensorflow as tf
model = tf.keras.Sequential([
    # tf.keras.layers.Input(60),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(26)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# %%
# training
import numpy as np
model.fit(np.array(X_train), np.array(y_train), epochs=10, batch_size=16)

# %%
# predict and evaluate