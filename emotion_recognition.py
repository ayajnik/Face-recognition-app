try:
    import sys, os
    import pandas as pd
    import numpy as np
    print('\n')
    print('Basic libraries imported for data manipulation.')
    print('\n')
except:
    print('some of the basic manipulation libraries are missing.')
try:
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten
    from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,AveragePooling2D
    from keras.losses import categorical_crossentropy
    from keras.optimizers import Adam
    from keras.regularizers import l2
    from keras.utils import np_utils
    print('\n')
    print('Deep Learning libraries imported.')
    print('\n')
except:
    print('Please review your Deep Learning packages as all packages not found.')

df=pd.read_csv('icml_face_data.csv')    ##importing the .csv file from kaggle. this csv file contains the pixels and is also labelled according to training and testing data. Every image is categorised with an emotion. A link will be provided for the dataset for more study

# print(df.info())
# print(df["Usage"].value_counts())

# print(df.head())
X_train,train_y,X_test,test_y=[],[],[],[]  ## stratifying our dataset

##Our next step is to :
##first, split the pixels according the white spaces as they are presented in a similar fashion in the dataset
##second, classify our images and data as per Training and Test data into the empty lists that we have created with their labels as well
for index, row in df.iterrows():
    val=row['pixels'].split(" ")
    try:
        if 'Training' in row['Usage']:
           X_train.append(np.array(val,'float32'))
           train_y.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
           X_test.append(np.array(val,'float32'))
           test_y.append(row['emotion'])
    except:
        print(f"error occured at index :{index} and row:{row}")

##defining our hyper-parameters for our cnn model
num_features = 64
num_labels = 7
batch_size = 64
epochs = 100
width, height = 48, 48

##we are converting our training and test data in float32 type data tyoe as our cnn accepts values in this format only. Although, we converted
##the values into float32 type data but we still have to convert the whole of our list into a float type data
X_train = np.array(X_train,'float32')
train_y = np.array(train_y,'float32')
X_test = np.array(X_test,'float32')
test_y = np.array(test_y,'float32')

##converting our labels into categorical values
#train_y=np_utils.to_categorical(train_y, num_classes=num_labels)
#test_y=np_utils.to_categorical(test_y, num_classes=num_labels)

###normalizing data between oand 1
X_train -= np.mean(X_train, axis=0)
X_train /= np.std(X_train, axis=0)

X_test -= np.mean(X_test, axis=0)
X_test /= np.std(X_test, axis=0)

## we have to reshape our data as per the size of the original data. as per our dataset, the pixels are in 48X48 size.
X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)

X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

##now building our convolutional neural network layer
#1st convolution layer
model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1:])))
model.add(Conv2D(64,kernel_size= (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.5))

#2nd convolution layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.5))

#3rd convolution layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

model.add(Flatten())

#fully connected neural networks
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_labels, activation='softmax'))

##we can find the summary of the model by running the next command. I am commenting it, but if you want to run, you can go ahead and
##uncomment it.

#model.summary()

#Compliling the model
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(),
              metrics=['accuracy'])

#Training the model
model.fit(X_train, train_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, test_y),
          shuffle=True)


#Saving the  model to  use it later on
fer_json = model.to_json()
with open("fer.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("fer.h5")