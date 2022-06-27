import random
from PIL import Image
from PIL import ImageEnhance

import matplotlib.pyplot as plt
def augment_img(image):
    # no hue gain since we don't want to change the actually color, only brightness to mimic lighting condition
    bright_gain = random.uniform(0.5, 2.0)  # saturation gain
    contrast_gain = random.uniform(0.5, 2.0)  # value gain

    # adjust the brightness
    img_b = ImageEnhance.Brightness(image).enhance(bright_gain)
    # adjust the contrast
    img_bc = ImageEnhance.Contrast(img_b).enhance(contrast_gain)

    return img_bc


data = Image.open('/data/Wanqing/YCB_Video_Dataset/data/0000/000001-color.png')
images = []
fig = plt.figure(figsize=(20, 20))
for i in range(16):
    images.append(augment_img(data))

for num, image in enumerate(images):
    fig.add_subplot(4, 4, num+1)
    plt.imshow(image)
plt.show()