#!/usr/bin/env python
# coding: utf-8

import numpy as np
from PIL import Image

if __name__ == '__main__':
    image_file = 'son.png'
    height = 100

    img = Image.open(image_file)
    img_width, img_height = img.size
    # print(img.size)
    
    width = 2 * height * img_width // img_height    # 假定字符的高度是宽度的2倍
    img = img.resize((width, height), Image.ANTIALIAS)
    
    pixels = np.array(img.convert('L'))
    # print(pixels.shape)
    # print(pixels)
    chars = "MNHQ$OC?7>!:-;. "
    N = len(chars)  # 16个字符
    step = 256 // N # 步长也是16
    # print(N)
    result = ''
    for i in range(height):
        for j in range(width):
            result += chars[pixels[i][j] // step]
        result += '\n'
    with open('text.txt', mode='w') as f:
        f.write(result)