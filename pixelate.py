import imageio as imio
import numpy as np

def pixelate(fname, a_in=256, a_out=64):
	img = imio.imread(fname).astype(np.float32)
	img_out = np.zeros_like(img)
	img_out_small = np.zeros([a_out, a_out, 3])

	k = a_in//a_out

	for i in range(a_out):
		for j in range(a_out):
			img_out[i*k:(i+1)*k, j*k:(j+1)*k] = np.mean(img[i*k:(i+1)*k, j*k:(j+1)*k], (0, 1))
			img_out_small[i, j] = np.mean(img[i*k:(i+1)*k, j*k:(j+1)*k], (0, 1))

	imio.imwrite('pixelate.png', img_out.astype(np.uint8))
	return img_out, img_out_small