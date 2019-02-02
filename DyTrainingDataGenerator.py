
# coding: utf-8


import numpy as np
import keras



class DySpeechGen(keras.utils.Sequence):
    def __init__(self, file_IDs, labels, batch_size=64, dim=16000, shuffle=False):
        self.Files = file_IDs
        self.labels = labels
        self.dim = dim
        self.batchSize = batch_size
        self.shuffle =shuffle
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.floor(len(self.Files)/self.batchSize))
    
    def load_all_data(self):
        indexes = self.indexes[:]
        # Find list of IDs
        list_IDs_temp = [self.Files[k] for k in indexes]
        #print("list_IDs_temp : ", list_IDs_temp)
        # Generate data
        X, y = self.__gen_part_training_data(list_IDs_temp)

            
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batchSize:(index+1)*self.batchSize]
        
        # Find list of IDs
        list_IDs_temp = [self.Files[k] for k in indexes]
        print("list_IDs_temp : ", list_IDs_temp)
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.allFilesLen = len(self.Files)
        self.indexes = np.arange(self.allFilesLen)
        #print("self.indexes is {}.",len(self.Files))
            
    def __data_generation(self,list_of_files):
        
        X = np.empty((self.batchSize,self.dim)) #64 files, each file is 16000 long
        y = np.empty((self.batchSize),dtype=int)
        
        #Start to generate the training data
        for i, _f in enumerate(list_of_files):
            print("current _f is : ",_f)
            #load npy file
            curX = np.load(_f)
            
            #check equal,smaller, or bigger
            #and truncate or padding
            #curX could be bigger or smaller than self.dim
            if curX.shape[0] == self.dim:
                X[i] = curX
                #print('Same dim')
            elif curX.shape[0] > self.dim: #bigger
                #we can choose any position in curX-self.dim
                randPos = np.random.randint(curX.shape[0]-self.dim) 
                X[i] = curX[randPos:randPos+self.dim]
                #print('File dim bigger')
            else: #smaller
                randPos = np.random.randint(self.dim-curX.shape[0])
                X[i,randPos:randPos+curX.shape[0]] = curX
                #print('File dim smaller')
        # Store class
            y[i] = self.labels[_f]

        return X, y
        

