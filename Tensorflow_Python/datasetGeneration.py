from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import os

# The batch size we'll use for training
batch_size = 64

# Size of the image required to train our model
img_size = 1024

# These many images will be used from the data archive
dataset_split = 2500

loadedNormalImages = [] # Container for storing the loaded normal images
# Loading the normal images
normal_dir = 'stone_normal'
for image_file in os.listdir( normal_dir )[ 0 : dataset_split ]:
    normal_image = Image.open( os.path.join( normal_dir , image_file ) ).resize( ( img_size , img_size ) )
    # Normalize the RGB image array
    normal_img_array = (np.asarray( normal_image ) ) / 255
    # Append both the image arrays
    loadedNormalImages.append( normal_img_array )

    
loadedPixelImages = [] # Container for storing the loaded pixelart images
pixel_dir = 'stone_pixel'
for image_file in os.listdir( pixel_dir )[ 0 : dataset_split ]:
    pixel_image = Image.open( os.path.join( pixel_dir , image_file ) ).resize( ( img_size , img_size ) )
    # Normalize the RGB image array
    pixel_img_array = (np.asarray( pixel_image ) ) / 255
    # Append both the image arrays
    loadedPixelImages.append( pixel_img_array )

# Train-test splitting
train_x, test_x, train_y, test_y = train_test_split( np.array(loadedNormalImages) , np.array(loadedPixelImages) , test_size=0.1 )

# Construct tf.data.Dataset object
dataset = tf.data.Dataset.from_tensor_slices( ( train_x , train_y ) )
dataset = dataset.batch( batch_size )

print(len(loadedNormalImages))
print(len(loadedPixelImages))