#coding:utf-8
import random
import os
from PIL import Image, ImageDraw, ImageFont

import CharactorSource

image_save_path = '/home/melt61/PictureGenerator/GenImage08'
label_save_path = '/home/melt61/PictureGenerator/GenLabel08/labels.txt'

 
font_path = '/home/melt61/PictureGenerator/OCR-Picture-Generators/fonts_Chinese'
font_dir = os.walk(font_path)
fonts=[]
for roo, dirs, files in font_dir:
    for f in files:
        fonts.append(os.path.join(roo, f))


image_id = 0
image_num = 10000

charactorInstant = CharactorSource.charactorsource()

print('class number',charactorInstant.map_length) 

while image_num > 0:
    gen_string = ''
    length = random.randint(5,10)
    length_i = length
    #image_size = 0
    
    while length_i > 0:
        mark_chi_selector = random.randint(0,99)
        if(mark_chi_selector > 30):
            chi_selector = random.randint(0, charactorInstant.chi_seed-1)
            gen_string = gen_string + charactorInstant.chi_list[chi_selector]
            #image_size = image_size +2
            
        elif(mark_chi_selector > 18):
            eng_0_selector = random.randint(0, charactorInstant.eng_0_seed-1)
            gen_string = gen_string + charactorInstant.eng_0_list[eng_0_selector]
            
        elif(mark_chi_selector > 10):
            eng_1_selector = random.randint(0, charactorInstant.eng_1_seed-1)
            gen_string = gen_string + charactorInstant.eng_1_list[eng_1_selector]
        
        elif(mark_chi_selector > 5):
            number_selector = random.randint(0, charactorInstant.number_seed-1)
            gen_string = gen_string + charactorInstant.number_list[number_selector]
        
        elif(mark_chi_selector > 2):
            half_mark_selector = random.randint(0, charactorInstant.half_mark_seed-1)
            gen_string = gen_string + charactorInstant.half_mark_list[half_mark_selector]
            #image_size = image_size +1
            
        else:
            full_mark_selector = random.randint(0, charactorInstant.full_mark_seed-1)
            gen_string = gen_string + charactorInstant.full_mark_list[full_mark_selector]
            #image_size = image_size +2
            
        length_i = length_i -1
    
    #print(gen_string)


    newIm = Image.new('L', (36*length, 50), 'white')
    ImDraw = ImageDraw.Draw(newIm)

    font_selector = random.randint(0,4)

    ttfont = ImageFont.truetype(fonts[font_selector], 36)
    
    ImDraw.text([0,0], gen_string, font = ttfont)
    
    newIm.save(image_save_path + '/'
                               + str(image_id) 
                               #+'_'
                               #+ gen_string 
                               + '.jpg', 'jpeg')
    
    image_num = image_num - 1 
    image_id = image_id + 1
    
    with open(label_save_path, 'a') as lf:
        lf.write(gen_string+'\n')
        
