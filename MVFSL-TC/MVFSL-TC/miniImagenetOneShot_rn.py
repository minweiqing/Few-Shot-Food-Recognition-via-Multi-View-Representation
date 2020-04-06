##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Albert Berenguel
## Computer Vision Center (CVC). Universitat Autonoma de Barcelona
## Email: aberenguel@cvc.uab.es
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#import torch
#import torch.utils.data as data
#import torchvision.transforms as transforms

from PIL import Image
import os.path
import csv
import math
import collections
from tqdm import tqdm
import pdb
import json

import numpy as np
# np.random.seed(2055)  # for reproducibility
np.random.seed(2191)
# LAMBDA FUNCTIONS
filenameToPILImage = lambda x: Image.open(x).convert('RGB')
PiLImageResize = lambda x: x.resize((224,224))

#class miniImagenetOneShotDataset(data.Dataset):
class FoodOneShotDataset():
    def __init__(self, dataroot = '/home/goodsrc/lyq_2/Food-101', type = 'train',
                 nEpisodes = 1000, classes_per_set=20, samples_per_class=1):

        self.nEpisodes = nEpisodes
        self.classes_per_set = classes_per_set
        self.samples_per_class = samples_per_class
        self.n_samples = self.samples_per_class * self.classes_per_set
        self.n_samplesNShot = 1 
        self.pointer = 0
        self.query_num=1
        if type == 'train':
            self.train = True
        else:
            self.train = False
        self.FoodImagesDir = os.path.join(dataroot,'images')
        self.data = json.load(open(os.path.join(dataroot,'meta',type + '_new.json'),'r'))
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
            # select n classes_per_set randomly
            selected_classes = np.random.choice(len(self.data.keys()), self.classes_per_set, False)
            #selected_class_meta_test = np.random.choice(selected_classes)
            support_set_x = []
            target_x = []
            for c in selected_classes:
                number_of_samples = self.samples_per_class + self.query_num
                selected_samples = np.random.choice(len(self.data[list(self.data.keys())[c]]),
                                                    number_of_samples, False)
                indexDtrain = np.array(selected_samples[:self.samples_per_class])
                support_set_x.append(np.array(self.data[list(self.data.keys())[c]])[indexDtrain].tolist())
                #if c == selected_class_meta_test:
                indexDtest = np.array(selected_samples[self.samples_per_class:])
                target_x.append(np.array(self.data[list(self.data.keys())[c]])[indexDtest].tolist())
            self.support_set_x_batch.append(support_set_x)
            self.target_x_batch.append(target_x)
    def get_train_item(self, index):
        support_set_x = np.zeros((self.n_samples, 224, 224, 3), dtype=np.float32)
        support_set_y = np.zeros((self.n_samples), dtype=np.int)
        target_x = np.zeros((self.classes_per_set*self.query_num, 224, 224, 3), dtype=np.float32)
        target_y = np.zeros((self.classes_per_set*self.query_num), dtype=np.int)
        #pdb.set_trace()
        flatten_support_set_x_batch = [os.path.join(self.FoodImagesDir, item)
                                       for sublist in self.support_set_x_batch[index] for item in sublist]
        support_set_y = np.array([self.classes_dict[item.split('/')[0]]
                                  for sublist in self.support_set_x_batch[index] for item in sublist])
        flatten_target_x = [os.path.join(self.FoodImagesDir, item)
                            for sublist in self.target_x_batch[index] for item in sublist]
        target_y = np.array([self.classes_dict[item.split('/')[0]]
                             for sublist in self.target_x_batch[index] for item in sublist])
        for i, path in enumerate(flatten_support_set_x_batch):
            tmp_img = filenameToPILImage(path.strip()+'.jpg')
            tmp_img = PiLImageResize(tmp_img)
            support_set_x[i] = tmp_img
        #
        for i, path in enumerate(flatten_target_x):
            tmp_img = filenameToPILImage(path.strip()+'.jpg')
            tmp_img = PiLImageResize(tmp_img)
            target_x[i] = tmp_img

        classes_dict_temp = {np.unique(support_set_y)[i]: i for i in np.arange(len(np.unique(support_set_y)))}
        support_set_y = np.array([classes_dict_temp[i] for i in support_set_y])
        target_y = np.array([classes_dict_temp[i] for i in target_y])
        return support_set_x, np.array(support_set_y), target_x, np.array(target_y)

    def __len__(self):
        return self.nEpisodes

    def get_batch(self, batch_size, shuffle = False):
        """
        Collects 1000 batches data for N-shot learning
        :param data_pack: Data pack to use (any one of train, val, test)
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        support_set_x = np.zeros((batch_size, self.classes_per_set, self.samples_per_class, 224, 224, 3), dtype=np.float32)
        support_set_y = np.zeros((batch_size, self.classes_per_set, self.samples_per_class), dtype=np.float32)
        if not self.train:
            target_x = np.zeros((batch_size, self.classes_per_set*self.query_num, 224, 224, 3), dtype=np.float32)
            target_y = np.zeros((batch_size, self.classes_per_set*self.query_num), dtype=np.float32)
        else:
            target_x = np.zeros((batch_size, self.classes_per_set*self.query_num, 224, 224, 3), dtype=np.float32)
            target_y = np.zeros((batch_size, self.classes_per_set*self.query_num), dtype=np.float32)
        if self.pointer >= self.__len__():
            self.pointer = 0
        if shuffle == True:
            choice_index = np.random.randint(0,self.__len__(),batch_size)
        else:
            choice_index = range(self.pointer, self.pointer + batch_size)
        self.pointer = self.pointer + batch_size
        for i in range(len(choice_index)):
            if self.train:
                one_support_set_x, one_support_set_y, one_target_x, one_target_y = self.get_train_item(choice_index[i])
            else:
                one_support_set_x, one_support_set_y, one_target_x, one_target_y = self.get_train_item(choice_index[i])
            support_set_x[i] = one_support_set_x.reshape(self.classes_per_set, self.samples_per_class, 224, 224, 3)
            support_set_y[i] = one_support_set_y.reshape(self.classes_per_set, self.samples_per_class)
            target_x[i] = one_target_x
            target_y[i] = one_target_y

        return np.squeeze(support_set_x, 0), np.squeeze(support_set_y, 0), np.squeeze(target_x), np.squeeze(target_y)

    def reset_pointer(self):
        self.pointer = 0