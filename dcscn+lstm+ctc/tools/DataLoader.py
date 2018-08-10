import numpy as np
from PIL import Image
import os
import re

from tools import CharactorSource

class data_loader:

    def __init__(self, input_path, label_path):

        self.image = self.load_images(input_path)
        self.label, self.label_att = self.load_labels(label_path)
    


    def get_batch_by_index(self, index = None):

        if index:
            image_batch = [self.image[i] for i in index]
            label_batch = [self.label[i] for i in index]
            label_batch_att = [self.label_att[i] for i in index]
        else:
            image_batch = self.image
            label_batch = self.label
            label_batch_att = self.label_att

        loaded_image = []

        for each_path in image_batch:
            with Image.open(each_path) as image_temp:
                image = np.asarray(image_temp, 'i')
                image = image.transpose(1, 0)
                loaded_image.append(image)

        padded_image = loaded_image 
        #self.input_padding(loaded_image)

        batch_seq_len = self.get_batch_input_length(padded_image)
        batch_label_len = self.get_batch_label_length(label_batch)
        batch_label_len_att = self.get_batch_label_att_length(label_batch_att)
        batch_sparse_label = self.label2sparse(label_batch)
        batch_flatten_label = self.getFlattenlabel(label_batch)
        batch_att_train_label = self.getAttTrainLabel(label_batch_att)
        

        #print(batch_sparse_label)

        return padded_image, batch_seq_len, batch_sparse_label, batch_label_len, batch_flatten_label, \
               label_batch_att, batch_att_train_label, batch_label_len_att

    def getAttTrainLabel(self, label):
        att_train = []
        for code in label:
            temp = np.concatenate(([0],code),0)
            att_train.append(temp)
        return att_train
    
    
    def get_batch_input_length(self,sequence):

        lengths = []
        for seq in sequence:
            lengths.append(seq.shape[0])

        return lengths

    def get_batch_label_length(self,sequence):

        lengths = []
        for seq in sequence:
            lengths.append(len(seq))
        
        #print(lengths)
        return lengths
    
    def get_batch_label_att_length(self,sequence):

        lengths = []
        for seq in sequence:
            lengths.append(len(seq)+2)
        
        #print(lengths)
        return lengths
        

    def load_images(self,path):

        image_save_path = path
        paths = []  

        for root, sub_dirs, files in os.walk(image_save_path):
            for special_file in files:
                special_file_path = os.path.join(root, special_file)
                paths.append(special_file_path)

        #print(paths[0])
        paths.sort(key = lambda i:int(re.search(r'/(\d+).jpg',i).group().lstrip('/').rstrip('.jpg')))
        #print(paths)

        #image = Image.open(paths[0])
        #image = np.asarray(image, 'i')
        #image.transpose(1,0)
        #image_array = np.array(image)

        return paths

    def load_labels(self, path):

        label_save_path = path
        label_data = list()

        with open(label_save_path) as label_file:
            for line in label_file.readlines():
                temp = line.rstrip('\n')
                label_data.append(temp)

        chr_ins = CharactorSource.charactorsource()
        #eng_ins = CharactorSource.charactorsource()
        sequences = list()
        sequences_att = list()

        for string in label_data:
            sequences.append(chr_ins.char2int(string))
            sequences_att.append(chr_ins.char2intAtt(string))
            #sequences.append(eng_ins.eng_char2int(string))
        
        #print(sequences)
        return sequences, sequences_att

    def label2sparse(self,sequence):

        indices = []
        values = []

        for index, seq in enumerate(sequence):
            indices.extend(zip([index] * len(seq), range(len(seq))))
            values.extend(seq)

        indices = np.asarray(indices, dtype=np.int32)
        values = np.asarray(values, dtype=np.int32)
        shape = np.asarray([len(sequence), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

        return indices, values, shape

    def getFlattenlabel(self,sequence):

        flattenLabel = [y for x in sequence for y in x]

        #print(flattenLabel)

        return flattenLabel

    def get_input_len(self):
        return len(self.image)

    def the_label(self, index):
        labels = []
        for i in index:
            labels.append(self.label[i])
        
        return labels

    def input_padding(self, input_batch):
        
        max_length = 0
        for seq in input_batch:
            if seq.shape[0] > max_length:
                max_length = seq.shape[0]
        
        padded_input = []

        for seq in input_batch:
            pad_length = max_length - seq.shape[0]
            while pad_length > 0:
                seq = np.vstack((seq, np.zeros(50)))
                pad_length = pad_length -1
            
            #seq = seq.reshape(max_length, 50, 1)
            padded_input.append(seq)
            #print(seq.shape)
    
        return padded_input
        



