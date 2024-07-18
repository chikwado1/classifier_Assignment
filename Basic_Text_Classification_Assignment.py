#!/usr/bin/env python
# coding: utf-8

# # Basic Text Classification Muticlass

# In[ ]:





# In[77]:


pip install tensorflow


# In[78]:


import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf


# In[101]:


# Update this path to the actual location of your dataset
dataset_dir = 'C:\\Users\\chikw\\Downloads\\stack_overflow_16k'

# List the contents of the dataset directory
print(os.listdir(dataset_dir))


# In[102]:


import os
import urllib.request
import zipfile

# Define the URL of the dataset and the target directory
url = 'https://example.com/path/to/stack_overflow_16k.zip'
target_dir = 'C:\\Users\\chikw\\Downloads\\stack_overflow_16k'

# Download the dataset
zip_path = os.path.join(target_dir, 'stack_overflow_16k.zip')
urllib.request.urlretrieve(url, zip_path)

# Extract the dataset
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(target_dir)

# Verify the extraction
print(os.listdir(target_dir))


# In[39]:


import os
import tarfile

# Path to the tar.gz file
archive_path = r"C:\Users\chikw\Downloads\stack_overflow_16k (3).tar.gz"

# Directory to extract the contents to
extract_dir = r"C:\Users\chikw\Downloads\stack_overflow_16k"

# Check if the archive exists
if not os.path.exists(archive_path):
    raise FileNotFoundError(f"The archive {archive_path} does not exist.")

# Extract the archive
with tarfile.open(archive_path, "r:gz") as tar:
    tar.extractall(path=extract_dir)

# Check if the extraction directory exists
if not os.path.exists(extract_dir):
    raise FileNotFoundError(f"The directory {extract_dir} does not exist after extraction.")

# List the contents of the extracted directory
files = os.listdir(extract_dir)
print(files)


# In[103]:


import tensorflow as tf
print(tf.__version__)


# In[104]:


import os

dataset_dir = '.\\stack_overflow_16k'  # or use the absolute path
if not os.path.exists(dataset_dir):
    raise FileNotFoundError(f"The directory {dataset_dir} does not exist.")
else:
    files = os.listdir(dataset_dir)
    print(files)


# In[41]:


from tensorflow.keras import layers
from tensorflow.keras import losses


# In[105]:


print(tf.__version__)


# In[43]:


url = "https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz"
dataset = tf.keras.utils.get_file("stack_overflow_16k", url,
                                    untar=True, cache_dir='.',
                                    cache_subdir='')

dataset_dir = os.path.join(os.path.dirname(dataset), 'stack_overflow_16k')


# In[44]:


os.listdir(dataset_dir)


# In[83]:


train_dir = os.path.join(dataset_dir, 'train')
os.listdir(train_dir)


# In[106]:


sample_file = os.path.join(train_dir, 'python/250.txt')
with open(sample_file) as f:
  print(f.read())


# In[107]:


batch_size = 32
seed = 42

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'stack_overflow_16k/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)


# In[108]:


for text_batch, label_batch in raw_train_ds.take(1):
  for i in range(3):
    print("Review", text_batch.numpy()[i])
    print("Label", label_batch.numpy()[i])
    


# In[87]:


print("Label 0 corresponds to", raw_train_ds.class_names[0])
print("Label 1 corresponds to", raw_train_ds.class_names[1])
print("Label 2 corresponds to", raw_train_ds.class_names[2])
print("Label 3 corresponds to", raw_train_ds.class_names[3])


# In[109]:


raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    'stack_overflow_16k/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)


# In[51]:


raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    'stack_overflow_16k/test',
    batch_size=batch_size)


# In[53]:


def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')


# In[54]:


max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)


# In[55]:


# Make a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)


# In[56]:


def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label


# In[110]:


# retrieve a batch (of 32 reviews and labels) from the dataset
text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print("Review", first_review)
print("Label", raw_train_ds.class_names[first_label])
print("Vectorized review", vectorize_text(first_review, first_label))


# In[90]:


print("1011 ---> ",vectorize_layer.get_vocabulary()[1011])
print(" 773 ---> ",vectorize_layer.get_vocabulary()[773])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))


# In[59]:


train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)


# In[60]:


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


# In[61]:


embedding_dim = 16


# In[111]:


model = tf.keras.Sequential([
  layers.Embedding(max_features, embedding_dim),
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(4, activation='sigmoid')])

model.summary()


# In[114]:


model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=['accuracy'])


# In[115]:


epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)


# In[65]:


loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)


# In[66]:


history_dict = history.history
history_dict.keys()


# In[116]:


acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[117]:


plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()


# In[118]:


import matplotlib.pyplot as plt

# Example data
epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
loss = [0.8, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15]
val_loss = [0.85, 0.65, 0.55, 0.5, 0.45, 0.4, 0.38, 0.35, 0.3, 0.28]

# Plotting the data
plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.show()


# In[119]:


import matplotlib.pyplot as plt

# Example data
epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
val_acc = [0.6, 0.65, 0.7, 0.72, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79]
val_loss = [0.85, 0.65, 0.55, 0.5, 0.45, 0.4, 0.38, 0.35, 0.3, 0.28]

fig, ax1 = plt.subplots()

# Plot validation accuracy
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Validation Accuracy', color='b')
ax1.plot(epochs, val_acc, 'bo-', label='Validation Accuracy')
ax1.tick_params(axis='y', labelcolor='b')

# Create a second y-axis for validation loss
ax2 = ax1.twinx()
ax2.set_ylabel('Validation Loss', color='r')
ax2.plot(epochs, val_loss, 'ro-', label='Validation Loss')
ax2.tick_params(axis='y', labelcolor='r')

# Adding titles and legends
fig.suptitle('Validation Accuracy and Loss')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.show()


# In[120]:


import matplotlib.pyplot as plt

# Example data
epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
train_loss = [0.8, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15]
val_loss = [0.85, 0.65, 0.55, 0.5, 0.45, 0.4, 0.38, 0.35, 0.3, 0.28]
val_acc = [0.6, 0.65, 0.7, 0.72, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79]

fig, ax1 = plt.subplots()

# Plot training loss and validation loss
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color='tab:red')
ax1.plot(epochs, train_loss, 'r--', label='Training Loss')
ax1.plot(epochs, val_loss, 'r-', label='Validation Loss')
ax1.tick_params(axis='y', labelcolor='tab:red')

# Create a second y-axis for validation accuracy
ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy', color='tab:blue')
ax2.plot(epochs, val_acc, 'b-', label='Validation Accuracy')
ax2.tick_params(axis='y', labelcolor='tab:blue')

# Adding titles and legends
fig.suptitle('Training Loss, Validation Loss, and Validation Accuracy')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.show()


# In[ ]:




