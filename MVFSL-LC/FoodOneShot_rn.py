from PIL import Image
import os.path
import csv
import math
import collections
from tqdm import tqdm
import pdb
import json
import scipy.io as sio

import numpy as np
np.random.seed(2191)  # for reproducibility
filenameToPILImage = lambda x: Image.open(x).convert('RGB')
PiLImageResize = lambda x: x.resize((84,84))
class FoodOneShotDataset():
    def __init__(self, dataroot = 'root_to_features',type = 'train',
                 nEpisodes = 1000, classes_per_set=5, samples_per_class=5):

        self.nEpisodes = nEpisodes
        self.classes_per_set = classes_per_set
        self.samples_per_class = samples_per_class
        self.n_samples = self.samples_per_class * self.classes_per_set
        self.n_samplesNShot = 1 
        self.pointer = 0
        self.query_num = 1
        if type == 'train':
            self.train = True
        else:
            self.train = False
        self.miniImagenetImagesDir = os.path.join(dataroot,'china-concat-'+type+ '_conv5_3')#root_to_feature(train or test)
        self.data = json.load(open(os.path.join('/media/goodsrc/d3f87cd5-e8d1-4aa1-b51f-a46bb097e6bb/goodsrc1/lyq/meta','chinafood-'+type + '_mat.json'),'r'))#root of json 
        self.data = collections.OrderedDict(sorted(self.data.items()))#store images name of each class
        self.classes_dict = {list(self.data.keys())[i]:i  for i in range(len(self.data.keys()))}#store length of each class
        if self.train:
            self.create_train_episodes(self.nEpisodes)
        else:
            self.create_train_episodes(self.nEpisodes)
    def create_train_episodes(self,episodes):
        self.support_set_x_batch = []
        self.target_x_batch = []
        for b in np.arange(episodes):
            selected_classes = np.random.choice(len(list(self.data.keys())), self.classes_per_set, False)
            support_set_x = []
            target_x = []
            for c in selected_classes:
                number_of_samples = self.samples_per_class +self.query_num
                selected_samples = np.random.choice(len(self.data[list(self.data.keys())[c]]),number_of_samples, False)
                indexDtrain = np.array(selected_samples[:self.samples_per_class])
                support_set_x.append(np.array(self.data[list(self.data.keys())[c]])[indexDtrain].tolist())
                indexDtest = np.array(selected_samples[self.samples_per_class:])
                target_x.append(np.array(self.data[list(self.data.keys())[c]])[indexDtest].tolist())
            self.support_set_x_batch.append(support_set_x)
            self.target_x_batch.append(target_x)
 
    def get_train_item(self, index):

        support_set_x = np.zeros((self.n_samples, 14,14,1024), dtype=np.float32)
        support_set_y = np.zeros((self.n_samples), dtype=np.int)
 
        target_x = np.zeros((self.classes_per_set*self.query_num, 14,14,1024), dtype=np.float32)
        target_y = np.zeros((self.classes_per_set*self.query_num), dtype=np.int)
        #pdb.set_trace()
        flatten_support_set_x_batch = [os.path.join(self.miniImagenetImagesDir, item.split('/')[1])
                                       for sublist in self.support_set_x_batch[index] for item in sublist]
        support_set_y = np.array([self.classes_dict[item.split('/')[0]]
                                  for sublist in self.support_set_x_batch[index] for item in sublist])
        flatten_target_x = [os.path.join(self.miniImagenetImagesDir, item.split('/')[1])
                            for sublist in self.target_x_batch[index] for item in sublist]
        target_y = np.array([self.classes_dict[item.split('/')[0]]
                             for sublist in self.target_x_batch[index] for item in sublist])
        for i, path in enumerate(flatten_support_set_x_batch):
            # print(path)
            data = sio.loadmat(path)
            data=data['feature']
            support_set_x[i] = data

        for i, path in enumerate(flatten_target_x):
            data = sio.loadmat(path)
            data= data['feature']
            target_x[i] = data

    
        classes_dict_temp = {np.unique(support_set_y)[i]: i for i in np.arange(len(np.unique(support_set_y)))}
        support_set_y = np.array([classes_dict_temp[i] for i in support_set_y])
        target_y = np.array([classes_dict_temp[i] for i in target_y])
    
        return support_set_x, np.array(support_set_y), target_x, np.array(target_y)

    def __len__(self):
        return self.nEpisodes
    def get_batch(self, batch_size, shuffle = False):
        support_set_x = np.zeros((batch_size, self.classes_per_set, self.samples_per_class, 14,14,1024), dtype=np.float32)
        support_set_y = np.zeros((batch_size, self.classes_per_set, self.samples_per_class), dtype=np.float32)
        if not self.train:
            target_x = np.zeros((batch_size, self.classes_per_set*self.query_num, 14,14,1024), dtype=np.float32)
            target_y = np.zeros((batch_size, self.classes_per_set*self.query_num), dtype=np.float32)
        else:
            target_x = np.zeros((batch_size, self.classes_per_set*self.query_num, 14,14,1024), dtype=np.float32)
            target_y = np.zeros((batch_size, self.classes_per_set*self.query_num), dtype=np.float32)
        if self.pointer >= self.__len__():
            self.pointer = 0
        if shuffle == True:
            choice_index = np.random.randint(0,self.__len__(),batch_size)
        else:
            choice_index = range(self.pointer, self.pointer + batch_size)
        self.pointer = self.pointer + batch_size
        for i in range(len(choice_index)):
            if not self.train:
                one_support_set_x, one_support_set_y, one_target_x, one_target_y = self.get_train_item(choice_index[i])
            else:
                one_support_set_x, one_support_set_y, one_target_x, one_target_y = self.get_train_item(choice_index[i])
            support_set_x[i] = one_support_set_x.reshape(self.classes_per_set, self.samples_per_class,14,14,1024)
            support_set_y[i] = one_support_set_y.reshape(self.classes_per_set, self.samples_per_class)
            target_x[i] = one_target_x
            target_y[i] = one_target_y

        return np.squeeze(support_set_x, 0), np.squeeze(support_set_y, 0), np.squeeze(target_x), np.squeeze(target_y)

    def reset_pointer(self):
        self.pointer = 0