import cv2
import numpy as np
from PIL import Image as img
from matplotlib import pyplot as plt

def rotate_image():
    img_path = '/Users/jessicapetrochuk/Documents/School/UBC/2021-2022/Directed Studies/Code/Detectron_2/myDATASET/images_with_rotations/section_img_0.jpg'
    image = img.open(img_path)
    image = image.rotate(angle=30)
    image_array = np.asarray(image)
    plt.imsave('hello.png', image_array)

if __name__ == '__main__':
    rotate_image()