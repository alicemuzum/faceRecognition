
import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
tf.enable_eager_execution()

# =============================================================================
# # Enable eager execution for getting length of dataset
# tf.compat.v1.enable_eager_execution()
# =============================================================================
 #Avoid Out Of Memory errorsby setting Gpu Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)
    
#Setup paths

POS_PATH = os.path.join('data','positive')
NEG_PATH = os.path.join('data','negative')
ANC_PATH = os.path.join('data','anchor')

# =============================================================================
# os.makedirs(POS_PATH)
# os.makedirs(NEG_PATH)
# os.makedirs(ANC_PATH)
# 
#  "!tar -xf lfw.tgz" for untar the lfw file
# 
# tüm fotoları bir dosyadan alıp diğer dosyaya attık
# for directory in os.listdir('lfw'):
#     for file in os.listdir(os.path.join('lfw', directory)):
#         EX_PATH = os.path.join('lfw',directory, file)
#         NEW_PATH = os.path.join(NEG_PATH,file)
#         os.replace(EX_PATH, NEW_PATH)
# =============================================================================

#for unique  image names
import uuid

# =============================================================================
# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     ret, frame = cap.read()
#     
#     #Cut down frame to 250x250px
#     frame = frame[120:120+250,200:200+250,:]
#     
#     #Collect anchors
#     if cv2.waitKey(1) & 0xFF == ord('a'):
#         #Create unique file path
#         imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
#         #Write out anchor image
#         cv2.imwrite(imgname,frame)
#     #Collect positives
#     if cv2.waitKey(1) & 0xFF == ord('p'):
#         #Create unique file path
#         imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
#         #Write out anchor image
#         cv2.imwrite(imgname,frame)
#     #Show image back to screen
#     cv2.imshow('Image Collection', frame)
#     
#     #Break
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# #Release the webcam
# cap.release()
# #Close the image show frame
# cv2.destroyAllWindows()
# =============================================================================

#--------------------------PREPROCESSING
anchor = tf.data.Dataset.list_files(ANC_PATH + '\*.jpg').take(300)
positive = tf.data.Dataset.list_files(POS_PATH + '\*.jpg').take(300)
negative = tf.data.Dataset.list_files(NEG_PATH + '\*.jpg').take(300)

def preprocess(file_path):
    
    # Read image from file path
    byte_img = tf.io.read_file(file_path)
    # Load the image
    img = tf.io.decode_jpeg(byte_img)
    
    # Preprocessing steps and resizing the iamge to be 100x100x3
    img = tf.image.resize(img, (100,100))
    img = img / 255.0
    return img

# Creating Labelled Dataset

positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(300))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(300))))
data = positives.concatenate(negatives)

def preprocessing_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)

# Build dataloader pipeline
data = data.map(preprocessing_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)

i = 0
for elem in data:
    i +=1
# Training Partition
train_data = data.take(round(i*.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

# Testing Partition
test_data = data.skip(round(i*.7))
test_data = test_data.take(round(i*.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)

# MODEL ENGINEERING#######################################

def make_embedding(): 
    inp = Input(shape=(100,100,3), name='input_image')
    
    # First block
    c1 = Conv2D(64, (10,10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2,2), padding='same')(c1)
    
    # Second block
    c2 = Conv2D(128, (7,7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
    
    # Third block 
    c3 = Conv2D(128, (4,4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2,2), padding='same')(c3)
    
    # Final embedding block
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    
    
    return Model(inputs=[inp], outputs=[d1], name='embedding')

embedding = make_embedding()

class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()
    
    def call(self,input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

def make_siamese_model():
    
    # Anchor image input in the netwprk
    input_image = Input(name='input_img', shape=(100,100,3))
    
    # Validation image in the network
    validation_image = Input(name='validation_img', shape=(100,100,3))
    
    inp_embedding = embedding(input_image)
    val_embedding = embedding(validation_image)
    
    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(inp_embedding, val_embedding)
    
    # Classification Layer 
    classifier = Dense(1, activation= 'sigmoid')(distances)
    
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

siamese_model = make_siamese_model()
    
##################### Training

# Setup Loss and Optimizer
binary_cross_loss = tf.keras.losses.BinaryCrossentropy()  
opt = tf.keras.optimizers.Adam(1e-4)#0.0001

# Checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir,'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

# Build Train Step Function

@tf.function
def train_step(batch):
    
    with tf.GradientTape() as tape:
        # Get anchor and positive img
        X = batch[:2]
        # Get label
        y = batch[2]
        
        # Forward pass
        yhat = siamese_model(X, training=True)
        # Calculate loss
        loss = binary_cross_loss(y,yhat)
    
    print(loss)
    #Calculate gradients
    grad = tape.gradient(loss,siamese_model.trainable_variables)
    
    #Calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad,siamese_model.trainable_variables))
    return loss

# Build Train Loop
def train(data, EPOCHS):
    #Loop through epochs
    for epoch in range(1,EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch,EPOCHS))
        progbar = tf.keras.utils.Progbar(td)
        
        #Loop through each batch
        for idx, batch in enumerate(data):
            #Run train step here
            train_step(batch)
            progbar.update(idx+1)
    
        # Save checkpoints
        if epoch % 10 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            
# Train the Model
EPOCHS = 50
td = 0
for i in train_data:
    td += 1
train(train_data,EPOCHS) 
 
############### TEST

from tensorflow.keras.metrics import Precision, Recall

# Get a batch of test data
test_input, test_val, y_true = test_data.as_numpy_iterator().next()

# Make predictions
y_hat = siamese_model.predict([test_input, test_val])

# Post processing the results
#[1 if prediction > 0.5 else 0 for prediction in y_hat]

# Creating a metric object
m = Recall()

#Calculating the recall value
m.update_state( y_true,y_hat)

# Return Recall Result
m.result().numpy()

plt.figure(figsize=(10,8))
plt.subplot(1,2,1)
plt.imshow(test_input[3])
plt.subplot(1,2,2)
plt.imshow(test_val[3])
plt.show()

#Save
siamese_model.save('siamesemodel.h5')

# Reload model
#model = tf.keras.model.load_model('siamesemodel.h5', custom_objects={'L1Dist':L1Dist, 'BinaryCrossEntropy':tf.losses.BinaryCrossentropy})