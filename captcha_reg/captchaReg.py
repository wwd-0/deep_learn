import tensorflow.compat.v1 as tf
from captcha.image  import ImageCaptcha
import numpy as np
from PIL import Image
import random
import sys
sys.path.append("../CNN/")

tf.disable_eager_execution()

def train_crack_captcha_cnn():
    cnn = tensorflow.CNN.CNN()
    cnn.conv([3,3,1,32])
    pass


def random_captcha_text(char_set=None,captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c  = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text

def gen_captcha_text_and_image(char_set):
    image = ImageCaptcha()
    captch_text = random_captcha_text(char_set)
    captch_text = ''.join(captch_text) #转成字符串
    captcha = image.generate(captch_text) #生成验证码图像

    captch_image = Image.open(captcha)
    captch_image = np.array(captch_image)
    return captch_text,captch_image

if __name__ == "__main__":
    train = 0
    if train    == 0:
        number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'g', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                    'u', 'v', 'w', 'x', 'y', 'z']
        ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'G', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                    'U', 'V', 'W', 'X', 'Y', 'Z']

        text,image = gen_captcha_text_and_image(number)
        print("验证码图像channel;",image.shape)
        #图像大小
        IAMGE_HEIGHT = 60
        IAMGE_WIDTH = 160
        MAX_CAPTCHA = len(text)
        print("验证码文本最长字符数:",MAX_CAPTCHA)
        char_set = number
        CHAR_SET_LEN = len(char_set)
        X = tf.placeholder(tf.float32,[None,IAMGE_HEIGHT*IAMGE_WIDTH])
        Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
        keep_prob = tf.placeholder(tf.float32) #dropout
        train_crack_captcha_cnn()

    if train == 1:
        number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        IAMGE_HEIGHT = 60
        IAMGE_WIDTH = 160
