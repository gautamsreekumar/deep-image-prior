import argparse
import os
import numpy as np

from model import dip
import tensorflow as tf

parser = argparse.ArgumentParser(description='')

parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=2, help='# images in batch')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--fname', dest='fname', default='image.jpg', help='input filename')
parser.add_argument('--task', dest='task', default='denoise', help='denoise/inpaint')

args = parser.parse_args()

def main(_):
	if not os.path.exists(args.checkpoint_dir):
		os.makedirs(args.checkpoint_dir)
	denoise = False
	inpaint = False
	with tf.Session() as sess:
		if args.task == 'denoise':
			denoise = True
		elif args.task == 'inpaint':
			inpaint = True
		
		model = dip(sess, fname=args.fname, batch_size=args.batch_size,
					epochs=args.epoch, save_freq=10, sample_freq=100,
					denoise=denoise, inpaint=inpaint)

		if args.phase == 'train':
			model.train_model()
		elif args.phase == 'test':
			pass
		elif args.phase == 'sample':
			model.sample_for_noises()
		else:
			print "Wrong choice for --phase argument"

if __name__ == '__main__':
	tf.app.run()
