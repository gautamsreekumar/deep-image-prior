import numpy as np
import scipy.misc as sp
import imageio as imio

def load(pos, img, l, stddev=1.0):
	img_x = []
	img_y = []
	for i in range(pos.shape[0]):
		temp_img = img[pos[i, 0]:pos[i, 0]+l, pos[i, 1]:pos[i, 1]+l].astype(np.float32)
		temp_noisy = np.clip(temp_img+np.random.normal(scale=stddev, size=temp_img.shape), 0, 255)
		img_x.append(temp_noisy/255.)
		img_y.append(temp_img/255.)
	return img_x, img_y

def load_test(imgs, stddev=1.0):
	img_x = []
	img_y = []
	for i in imgs:
		temp_img = imio.imread('./test_images/{}.jpg'.format(i+1)).astype(np.float32)
		temp_noisy = np.clip(temp_img+np.random.normal(scale=stddev, size=temp_img.shape), 0, 255)
		img_x.append(temp_noisy/255.)
		img_y.append(temp_img/255.)
	return img_x, img_y