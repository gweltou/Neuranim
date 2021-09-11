#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image
import os
import numpy as np

imglist = [os.path.join("frames", fname) for fname in os.listdir("frames")]

sumimg = np.asarray(Image.open(imglist[0]))
sumimg = sumimg.astype('uint32')

for i in imglist[1:]:
    temp = np.asarray(Image.open(i))
    temp = temp.astype('uint32')
    sumimg += temp

avg = sumimg / (len(imglist) / 1)
#avg = np.clip
final = Image.fromarray(avg.astype('uint8'))

final.save("final.png")
