#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Tran Minh Quan, quantm@unist.ac.kr
import os, sys, argparse, glob, cv2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Misc. libraries
from six.moves import map, zip, range
from natsort import natsorted 

# Array and image processing toolboxes
import numpy as np 
import skimage
import skimage.io
import skimage.transform

###################################################################################################
from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.tfutils import optimizer, gradproc
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.utils import logger


###################################################################################################
EPOCH_SIZE = 100
NB_FILTERS = 32	  # channel size

DIMX  = 256
DIMY  = 256
DIMZ  = 256
DIMC  = 1
####################################################################################################
class ImageDataFlow(RNGDataFlow):
	def __init__(self, image_path, style_path, size, dtype='float32', isTrain=False, isValid=False):
		self.dtype      	= dtype
		self.image_path   	= image_path
		self.style_path   	= style_path
		self._size      	= size
		self.isTrain    	= isTrain
		self.isValid    	= isValid

	def size(self):
		return self._size

	def reset_state(self):
		self.rng = get_rng(self)

	def get_data(self, shuffle=True):
		#
		# Read and store into pairs of images and labels
		#
		images = glob.glob(self.image_path + '/*.*')
		styles = glob.glob(self.style_path + '/*.*')

		if self._size==None:
			self._size = len(images)

		from natsort import natsorted
		images = natsorted(images)
		styles = natsorted(styles)


		#
		# Pick the image over size 
		#
		for k in range(self._size):
			#
			# Pick randomly a tuple of training instance
			#
			rand_image = np.random.randint(0, len(images))
			rand_style = np.random.randint(0, len(styles))

			if self.isTrain:
				# Read the 3D image and style
				image = skimage.io.imread(images[rand_image])
				style = skimage.io.imread(styles[rand_style])

				# Resize if necessary 
				# style = skimage.transform.resize

				# Rotate and resample volume
				import scipy.ndimage.interpolation
				degrees = np.random.uniform(low=0.0, high=360.0)

				image = scipy.ndimage.interpolation.rotate(image, 
					angle=degrees, 
					axes=(1, 2), 
					reshape=False, #If reshape is true, the output shape is adapted so that the input 
					#array is contained completely in the output. Default is True
					order=3, mode='reflect')

				else:
					# Adjust via callback 
					# Rotate 360 degree
					pass
				yield [image.astype(np.float32), 
				   	   style.astype(np.float32), 
				   	   ]
####################################################################################################
def get_data(image_path, style_path, size=EPOCH_SIZE):
	ds_train = ImageDataFlow(image_path,
							 style_path, 
							 size, 
							 isTrain=True
							 )

	ds_valid = ImageDataFlow(image_path,
							 style_path, 
							 size, 
							 isValid=True
							 )

	ds_train.reset_state()
	ds_valid.reset_state() 

	return ds_train, ds_valid

			
####################################################################################################
def INReLU(x, name=None):
	x = InstanceNorm('inorm', x)
	return tf.nn.relu(x, name=name)


def INLReLU(x, name=None):
	x = InstanceNorm('inorm', x)
	return tf.nn.leaky_relu(x, name=name)
	
def BNLReLU(x, name=None):
	x = BatchNorm('bn', x)
	return tf.nn.leaky_relu(x, name=name)

###############################################################################
# Utility function for scaling 
def tf_2tanh(x, maxVal = 255.0, name='ToRangeTanh'):
	with tf.variable_scope(name):
		return (x / maxVal - 0.5) * 2.0
###############################################################################
def tf_2imag(x, maxVal = 255.0, name='ToRangeImag'):
	with tf.variable_scope(name):
		return (x / 2.0 + 0.5) * maxVal

# Utility function for scaling 
def np_2tanh(x, maxVal = 255.0, name='ToRangeTanh'):
	return (x / maxVal - 0.5) * 2.0
###############################################################################
def np_2imag(x, maxVal = 255.0, name='ToRangeImag'):
	return (x / 2.0 + 0.5) * maxVal

###############################################################################
# FusionNet
@layer_register(log_shape=True)
def residual(x, chan, first=False):
	with argscope([Conv2D], nl=INLReLU, stride=1, kernel_shape=3):
		input = x
		return (LinearWrap(x)
				.Conv2D('conv1', chan, padding='SAME', dilation_rate=1)
				.Conv2D('conv2', chan, padding='SAME', dilation_rate=2)
				.Conv2D('conv4', chan, padding='SAME', dilation_rate=4)				
				.Conv2D('conv5', chan, padding='SAME', dilation_rate=8)
				.Conv2D('conv0', chan, padding='SAME', nl=tf.identity)
				.InstanceNorm('inorm')()) + input

###############################################################################
@layer_register(log_shape=True)
def Subpix2D(inputs, chan, scale=2, stride=1):
	with argscope([Conv2D], nl=INLReLU, stride=stride, kernel_shape=3):
		results = Conv2D('conv0', inputs, chan* scale**2, padding='SAME')
		old_shape = inputs.get_shape().as_list()
		# results = tf.reshape(results, [-1, chan, old_shape[2]*scale, old_shape[3]*scale])
		# results = tf.reshape(results, [-1, old_shape[1]*scale, old_shape[2]*scale, chan])
		if scale>1:
			results = tf.depth_to_space(results, scale, name='depth2space', data_format='NHWC')
		return results

###############################################################################
@layer_register(log_shape=True)
def residual_enc(x, chan, first=False):
	with argscope([Conv2D, Deconv2D], nl=INLReLU, stride=1, kernel_shape=3):
		x = (LinearWrap(x)
			# .Dropout('drop', 0.75)
			.Conv2D('conv_i', chan, stride=2) 
			.residual('res_', chan, first=True)
			.Conv2D('conv_o', chan, stride=1) 
			())
		return x

###############################################################################
@layer_register(log_shape=True)
def residual_dec(x, chan, first=False):
	with argscope([Conv2D, Deconv2D], nl=INLReLU, stride=1, kernel_shape=3):
				
		x = (LinearWrap(x)
			.Subpix2D('deconv_i', chan, scale=1) 
			.residual('res2_', chan, first=True)
			.Subpix2D('deconv_o', chan, scale=2) 
			# .Dropout('drop', 0.75)
			())
		return x

###############################################################################
@auto_reuse_variable_scope
def arch_generator(img, last_dim=1):
	assert img is not None
	with argscope([Conv2D, Deconv2D], nl=INLReLU, kernel_shape=3, stride=2, padding='SAME'):
		e0 = residual_enc('e0', img, NB_FILTERS*1)
		e1 = residual_enc('e1',  e0, NB_FILTERS*2)
		e2 = residual_enc('e2',  e1, NB_FILTERS*4)

		e3 = residual_enc('e3',  e2, NB_FILTERS*8)
		# e3 = Dropout('dr', e3, 0.5)

		d3 = residual_dec('d3',    e3, NB_FILTERS*4)
		d2 = residual_dec('d2', d3+e2, NB_FILTERS*2)
		d1 = residual_dec('d1', d2+e1, NB_FILTERS*1)
		d0 = residual_dec('d0', d1+e0, NB_FILTERS*1) 
		dd =  (LinearWrap(d0)
				.Conv2D('convlast', last_dim, kernel_shape=3, stride=1, padding='SAME', nl=tf.tanh, use_bias=True) ())
		return dd

####################################################################################################
class Model(ModelDesc):
	def _get_inputs(self):
		pass

	def _build_graph(self, inputs):
		pass

	def _get_optimizer(self):
		pass

###################################################################################################
def apply(model_path, image_path, style_path):
	pass

###################################################################################################
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', 	help='comma separated list of GPU(s) to use.')
	parser.add_argument('--load', 	help='load model')
	parser.add_argument('--apply', 	action='store_true')
	parser.add_argument('--image', 	help='path to the image. ', default="data/image")
	parser.add_argument('--style',  help='path to the style. ', default="data/style")
	parser.add_argument('--vgg19', 	help='load model', 			default="data/vgg19.npz")
	parser.add_argument('--output', help='directory for saving the rendering', default=".", type=str)
	args = parser.parse_args()

	if args.gpu:
		os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

	if args.apply:
		apply(args.load, args.image, args.style)
	else:
		# Set the logger directory
		logger.auto_set_dir()

		nr_tower = max(get_nr_gpu(), 1)
		ds_train = QueueInput(get_data(args.data))
		model = Model()

		if args.load:
            session_init = SaverRestore(args.load)
        else:
            assert os.path.isfile(args.vgg19)
            param_dict = dict(np.load(args.vgg19))
            param_dict = {'VGG19/' + name: value for name, value in six.iteritems(param_dict)}
            session_init = DictRestore(param_dict)

		
			# Set up configuration
			config = TrainConfig(
				model           =   model, 
				dataflow        =   ds_train,
				callbacks       =   [
					PeriodicTrigger(ModelSaver(), every_k_epochs=50),
					PeriodicTrigger(VisualizeRunner(ds_valid), every_k_epochs=5),
					#PeriodicTrigger(InferenceRunner(ds_valid, [ScalarStats('loss_membr')]), every_k_epochs=5),
					ScheduledHyperParamSetter('learning_rate', [(0, 2e-4), (100, 1e-4), (200, 1e-5), (300, 1e-6)], interp='linear')
					#ScheduledHyperParamSetter('learning_rate', [(30, 6e-6), (45, 1e-6), (60, 8e-7)]),
	            	#HumanHyperParamSetter('learning_rate'),
					],
				max_epoch       =   500, 
				session_init    =   session_init,
				nr_tower        =   max(get_nr_gpu(), 1)
				)
		
			# Train the model
			SyncMultiGPUTrainer(config).train()

	
	
		