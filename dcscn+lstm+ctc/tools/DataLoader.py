import numpy as np
from PIL import Image
import os
import re

from tools import CharactorSource

class data_loader:

    def __init__(self, input_path, label_path):

        self.image = self.load_images(input_path)
        self.label = self.load_labels(label_path)


    def get_batch_by_index(self, index = None):

        if index:
            image_batch = [self.image[i] for i in index]
            label_batch = [self.label[i] for i in index]
        else:
            image_batch = self.image
            label_batch = self.label

        loaded_image = []

        for each_path in image_batch:
            with Image.open(each_path) as image_temp:
                image = np.asarray(image_temp, 'i')
                image = image.transpose(1, 0)
                loaded_image.append(image)

        padded_image = self.input_padding(loaded_image)

        batch_seq_len = self.get_batch_input_length(padded_image)
        batch_sparse_label = self.label2sparse(label_batch)

        #print(batch_sparse_label)

        return padded_image, batch_seq_len, batch_sparse_label

    def get_batch_input_length(self,sequence):

        lengths = []
        for seq in sequence:
            lengths.append(seq.shape[0])

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

        for string in label_data:
            sequences.append(chr_ins.char2int(string))
            #sequences.append(eng_ins.eng_char2int(string))
        
        #print(sequences)
        return sequences

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
            
            padded_input.append(seq)
            #print(seq.shape)
    
        return padded_input
        



