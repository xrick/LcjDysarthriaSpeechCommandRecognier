#import tensorflow as tf
import librosa
import os
import numpy as np
from tqdm import tqdm


DATA_PATH = "../DeadSimpleSpeechRecognizor_Data/"
DYData_Path = "../DysSpeech_Corpora/Linzy/linzyCut/train/"
DYNPY_PATH = "./mfcc"
DysTrainedModelPath = "trained_models"

# Input: Folder Path
# Output: Tuple (Label, Indices of the labels, one-hot encoded labels)
def get_labels(path=DATA_PATH):
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)


# Handy function to convert wav2mfcc
def wav2mfcc(file_path, max_len=11):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=16000)

    # If maximum length exceeds mfcc lengths then pad the remaining ones
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Else cutoff the remaining parts
    else:
        mfcc = mfcc[:, :max_len]
    
    return mfcc


def save_data_to_array(path=DATA_PATH, max_len=11):
    labels, _, _ = get_labels(path)
    print("labels are :",labels)
    for label in labels:
        # Init mfcc vectors
        mfcc_vectors = []

        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
            mfcc = wav2mfcc(wavfile, max_len=max_len)
            mfcc_vectors.append(mfcc)
        print("length of mfcc_vectors is : ",len(mfcc))
        np.save(label + '.npy', mfcc_vectors)

'''
def get_train_test(split_ratio=0.6, random_state=42):
    # Get available labels
    labels, indices, _ = get_labels(DATA_PATH)

    # Getting first arrays
    X = np.load(labels[0] + '.npy')
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load(label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

    assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size= (1 - split_ratio), random_state=random_state, shuffle=True)
'''

def prepare_dataset(path=DATA_PATH):
    labels, _, _ = get_labels(path)
    data = {}
    for label in labels:
        data[label] = {}
        data[label]['path'] = [path  + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]

        vectors = []

        for wavfile in data[label]['path']:
            wave, sr = librosa.load(wavfile, mono=True, sr=None)
            # Downsampling
            wave = wave[::3]
            mfcc = librosa.feature.mfcc(wave, sr=16000)
            vectors.append(mfcc)

        data[label]['mfcc'] = vectors

    return data


def load_dataset(path=DATA_PATH):
    data = prepare_dataset(path)

    dataset = []

    for key in data:
        for mfcc in data[key]['mfcc']:
            dataset.append((key, mfcc))

    return dataset[:100]

def get_train_set():
    labels, indices, _ = get_labels(DYNPY_PATH)
    # Getting first arrays
    print("labels[0].npy is {} ".format(labels[0]))
    print("labels[1:] : {}".format(labels[1:]))

    X = np.load(os.path.join(DYNPY_PATH,labels[0]))
    y = np.zeros(X.shape[0])
    #print("X.shape is {}".format(X.shape))

    for i, label in enumerate(labels[1:]):
        x = np.load(os.path.join(DYNPY_PATH, label))
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value=(i + 1)))

    assert X.shape[0] == len(y)
    print("X.shape[0] is {}".format(X.shape))
    print("length y is {}".format(len(y)))
    #return train_test_split(X, y, test_size=(1 - split_ratio), random_state=random_state, shuffle=True)
    return (X,y)

#import SpeechModels
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
#from keras.optimizers import adadelta
num_of_classes = 19
feature_dim_1 = 20
feature_dim_2 = 11
channel = 1
def run_main():
    X_train, y_train = get_train_set()
    #reshape X_train
    X_train = X_train.reshape(X_train.shape[0],20,11,1)
    print("X_train shape is {}".format(X_train.shape))
    y_train_hot = to_categorical(y_train)
    print(y_train)
    print("Start to train.......")
    #adadelta = adadelta(lr=0.000001)
    '''
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu',
                     input_shape=(feature_dim_1, feature_dim_2, channel)))
    model.add(Conv2D(48, kernel_size=(2, 2), activation='relu'))
    model.add(Conv2D(120, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_of_classes, activation='softmax'))
    '''
    model = Sequential()  
    model.add(Conv2D(32, kernel_size=(2, 2),
                 activation='relu', input_shape=(20, 11, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.25))
    model.add(Dense(num_of_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.adadelta(lr=0.000000001),
                  metrics=['accuracy'])
    model.fit(X_train, y_train_hot, batch_size=64, epochs=300)
    print("Saving Model.........")
    #model.save('AttRNN_model.h5')
    model.save(os.path.join('.', DysTrainedModelPath, 'CNN2_model.h5'))

# print(prepare_dataset(DATA_PATH))

if __name__ == "__main__":
    run_main()
