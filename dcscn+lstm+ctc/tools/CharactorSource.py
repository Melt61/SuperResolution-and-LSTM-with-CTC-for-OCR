#coding:utf-8
import random
import os

class charactorsource(object):

    def __init__(self):
        self.half_mark_list = [',', '.', ';', 
                  '\'', '[', ']', '\\',
                  '<', '>', '?', ':', 
                  '\"', '{', '}', '|',
                  '`', '~', '/', ' ',
                  '!', '@', '#', '$',
                  '%', '^', '&', '*',
                  '(', ')', '-', '_',
                  '+', '=']
                  
        self.half_mark_seed = len(self.half_mark_list)

        self.full_mark_list = ['，', '。', '；', '’', '‘', 
                 '·', '「', '」','、', 
                 '～', '！','￥', '…',
                  '×', '（', '）', '—',
                 '《', '》', "？", '：', 
                 '”', '“']
        self.full_mark_seed = len(self.full_mark_list)


        self.eng_0_list = ['q', 'w', 'e', 'r', 't', 'y',
              'u', 'i', 'o', 'p', 'a', 's',
              'd', 'f', 'g', 'h', 'j', 'k',
              'l', 'z', 'x', 'c', 'v', 'b',
              'n', 'm']
        self.eng_0_seed = len(self.eng_0_list)

        self.eng_1_list = ['Q', 'W', 'E', 'R', 'T', 'Y',
              'U', 'I', 'O', 'P', 'A', 'S',
              'D', 'F', 'G', 'H', 'J', 'K',
              'L', 'Z', 'X', 'C', 'V', 'B',
              'N', 'M']
        self.eng_1_seed = len(self.eng_1_list)

        self.number_list = ['0', '1', '2', '3', '4', '5',
               '6', '7', '8', '9']
        self.number_seed = len(self.number_list)

        #i = 0
        self.chi_list = []
        with open('/home/melt61/PictureGenerator/OCR-Picture-Generators/Chinese-Generation/common/dict-common') as f:
            for line in f.readlines():
                #i = i+1
                temp = line.rstrip('\n')
                self.chi_list.append(temp)
                #print(line)

        self.chi_seed = len(self.chi_list)

        self.map_length = self.half_mark_seed + self.full_mark_seed + self.eng_0_seed + self.eng_1_seed + self.number_seed + self.chi_seed

        self.charactor_map_list = self.half_mark_list + self.full_mark_list + self.eng_0_list + self.eng_1_list + self.number_list + self.chi_list

        self.eng_map_length = self.half_mark_seed + self.eng_0_seed + self.eng_1_seed + self.number_seed
        self.eng_charactor_map_list = self.half_mark_list + self.eng_0_list + self.eng_1_list + self.number_list

        self.char_to_int = dict((c, i) for i, c in enumerate(self.charactor_map_list))
        self.int_to_char = dict((i, c) for i, c in enumerate(self.charactor_map_list))

        self.eng_char_to_int = dict((c,i) for i,c in enumerate(self.eng_charactor_map_list))
        self.eng_int_to_char = dict((i,c) for i,c in enumerate(self.eng_charactor_map_list))

    def str2one_hot(self,data):
        integer_encoded = [self.char_to_int[char] for char in data]
        #print(integer_encoded)
        # one hot encode
        onehot_encoded = list()
        for value in integer_encoded:
            letter = [0 for _ in range(len(self.charactor_map_list))]
            letter[value] = 1
            onehot_encoded.append(letter)
        #print(onehot_encoded)
        return onehot_encoded

    def char2int(self, data):
        integer_encoded = [self.char_to_int[char] for char in data]
        return integer_encoded

    def int2char(self, data):
        integer_decoded = [self.int_to_char[index] for index in data]
        return integer_decoded

    def eng_char2int(self, data):
        integer_encoded = [self.eng_char_to_int[char] for char in data]
        return integer_encoded

    def eng_int2char(self, data):
        integer_decoded = [self.eng_int_to_char[index] for index in data]
        return integer_decoded