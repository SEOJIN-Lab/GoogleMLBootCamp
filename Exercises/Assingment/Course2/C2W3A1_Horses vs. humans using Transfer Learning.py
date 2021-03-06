# Import all the necessary files!
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model

# Download the inception v3 weights
#!wget - -no - check - certificate \
 #   https: // storage.googleapis.com / mledu - datasets / inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
  #            - O / tmp / inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5

# Import the inception model
from tensorflow.keras.applications.inception_v3 import InceptionV3

# Create an instance of the inception model from the local pre-trained weights
local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape=(150, 150, 3),  ### YOUR CODE HERE
                                include_top=False,
                                weights=None)

pre_trained_model.load_weights(local_weights_file)

# Make all the layers in the pre-trained model non-trainable
for layer in pre_trained_model.layers:
    layer.trainable = False  ### YOUR CODE HERE

# Print the model summary
pre_trained_model.summary()

# Expected Output is extremely large, but should end with:

# batch_normalization_v1_281 (Bat (None, 3, 3, 192)    576         conv2d_281[0][0]
# __________________________________________________________________________________________________
# activation_273 (Activation)     (None, 3, 3, 320)    0           batch_normalization_v1_273[0][0]
# __________________________________________________________________________________________________
# mixed9_1 (Concatenate)          (None, 3, 3, 768)    0           activation_275[0][0]
#                                                                 activation_276[0][0]
# __________________________________________________________________________________________________
# concatenate_5 (Concatenate)     (None, 3, 3, 768)    0           activation_279[0][0]
#                                                                 activation_280[0][0]
# __________________________________________________________________________________________________
# activation_281 (Activation)     (None, 3, 3, 192)    0           batch_normalization_v1_281[0][0]
# __________________________________________________________________________________________________
# mixed10 (Concatenate)           (None, 3, 3, 2048)   0           activation_273[0][0]
#                                                                 mixed9_1[0][0]
#                                                                 concatenate_5[0][0]
#                                                                 activation_281[0][0]
# ==================================================================================================
# Total params: 21,802,784
# Trainable params: 0
# Non-trainable params: 21,802,784

last_layer = pre_trained_model.get_layer('mixed7') ### YOUR CODE HERE)
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output ### YOUR CODE HERE

# Expected Output:
# ('last layer output shape: ', (None, 7, 7, 768))

# Define a Callback class that stops training once accuracy reaches 99.9%
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.999):
      print("\nReached 99.9% accuracy so cancelling training!")
      self.model.stop_training = True

from tensorflow.keras.optimizers import RMSprop

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)### YOUR CODE HERE)(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)### YOUR CODE HERE)(x)
# Add a final sigmoid layer for classification
x = layers.Dense  (1, activation='sigmoid')(x)### YOUR CODE HERE)(x)

model = Model( pre_trained_model.input, x) ### YOUR CODE HERE, x)

model.compile(optimizer = RMSprop(learning_rate=0.0001),
              loss = 'binary_crossentropy', ### YOUR CODE HERE,
              metrics = ['accuracy'] ### YOUR CODE HERE,
              )

model.summary()

# Expected output will be large. Last few lines should be:

# mixed7 (Concatenate)            (None, 7, 7, 768)    0           activation_248[0][0]
#                                                                  activation_251[0][0]
#                                                                  activation_256[0][0]
#                                                                  activation_257[0][0]
# __________________________________________________________________________________________________
# flatten_4 (Flatten)             (None, 37632)        0           mixed7[0][0]
# __________________________________________________________________________________________________
# dense_8 (Dense)                 (None, 1024)         38536192    flatten_4[0][0]
# __________________________________________________________________________________________________
# dropout_4 (Dropout)             (None, 1024)         0           dense_8[0][0]
# __________________________________________________________________________________________________
# dense_9 (Dense)                 (None, 1)            1025        dropout_4[0][0]
# ==================================================================================================
# Total params: 47,512,481
# Trainable params: 38,537,217
# Non-trainable params: 8,975,264

# Get the Horse or Human dataset
#!gdown - -id 1onaG42NZft3wCE1WH0GDEbUhu75fedP5

# Get the Horse or Human Validation dataset
#!gdown - -id 1LYeusSEIiZQpwN-mthh5nKdA75VsKG1U

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import zipfile

test_local_zip = './horse-or-human.zip'
zip_ref = zipfile.ZipFile(test_local_zip, 'r')
zip_ref.extractall('./training')

val_local_zip = './validation-horse-or-human.zip'
zip_ref = zipfile.ZipFile(val_local_zip, 'r')
zip_ref.extractall('./validation')

zip_ref.close()

# Define our example directories and files
train_dir = './training'
validation_dir = './validation'

# Directory with our training horse pictures
train_horses_dir = os.path.join(train_dir, 'horses') ### YOUR CODE HERE
# Directory with our training humans pictures
train_humans_dir = os.path.join(train_dir, 'humans') ### YOUR CODE HERE
# Directory with our validation horse pictures
validation_horses_dir = os.path.join(validation_dir, 'horses') ### YOUR CODE HERE
# Directory with our validation humanas pictures
validation_humans_dir = os.path.join(validation_dir, 'humans') ### YOUR CODE HERE

train_horses_fnames = os.listdir(train_horses_dir) ### YOUR CODE HERE)
train_humans_fnames = os.listdir(train_humans_dir) ### YOUR CODE HERE)
validation_horses_fnames = os.listdir(validation_horses_dir) ### YOUR CODE HERE)
validation_humans_fnames = os.listdir(validation_humans_dir) ### YOUR CODE HERE)

print(len(train_horses_fnames)) ### YOUR CODE HERE)
print(len(train_humans_fnames)) ### YOUR CODE HERE)
print(len(validation_horses_fnames)) ### YOUR CODE HERE)
print(len(validation_humans_fnames)) ### YOUR CODE HERE)

# Expected Output:
# 500
# 527
# 128
# 128

# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255., ### YOUR CODE HERE)
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator( rescale = 1.0/255. ) ### YOUR CODE HERE)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir, ### YOUR CODE HERE)
                                                    batch_size = 20,
                                                    class_mode = 'binary',
                                                    target_size = (150, 150))

# Flow validation images in batches of 20 using test_datagen generator
validation_generator =  test_datagen.flow_from_directory( validation_dir, ### YOUR CODE HERE)
                                                          batch_size  = 20,
                                                          class_mode  = 'binary',
                                                          target_size = (150, 150))

# Expected Output:
# Found 1027 images belonging to 2 classes.
# Found 256 images belonging to 2 classes.

# Run this and see how many epochs it should take before the callback
# fires, and stops training at 99.9% accuracy
# (It should take less than 100 epochs)
callbacks = myCallback() ### YOUR CODE HERE)
history = model.fit( ### YOUR CODE HERE)
            train_generator,
            validation_data = validation_generator,
            steps_per_epoch = 100,
            epochs = 100,
            validation_steps = 5,
            verbose = 2,
            callbacks=callbacks)

import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()