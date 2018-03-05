#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Tran Minh Quan, quantm@unist.ac.kr
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os, sys, argparse, glob, cv2, six



# Misc. libraries
from six.moves import map, zip, range
from natsort import natsorted 

# Array and image processing toolboxes
import numpy as np 
import skimage
import skimage.io
import skimage.color
import skimage.transform

import tensorflow as tf
###################################################################################################
from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.utils.utils import get_rng
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

		# print(images)
		# print(styles)

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
				# Read the 3D image
				image = skimage.io.imread(images[rand_image])
				if image.shape != [DIMZ, DIMY, DIMX]: # Pad the image
					dimz, dimy, dimx = image.shape
					patz, paty, patx = (DIMZ-dimz)/2, (DIMY-dimy)/2, (DIMX-dimx)/2
					patz, paty, patx = int(patz), int(paty), int(patx)
					image = np.pad(image, ((patz, patz), (paty, paty), (patx, patx)), 
								   mode='constant', 
								   constant_values=0, 
						)
				image = np.transpose(image, [1, 2, 0])
				# image = np.expand_dims(image, axis=-1)
				image = np.expand_dims(image, axis=0)

				# Read the style
				style = skimage.io.imread(styles[rand_style])
				if style.ndim == 2: # If gray image, convert to 3 channel
					style = skimage.color.gray2rgb(style)
					# style = cv2.cvtColor(style, cv2.GRAY2RGB)
				seed = np.random.randint(0, 20152015)
				style = self.random_flip(style, seed=seed)        
				style = self.random_reverse(style, seed=seed)
				style = self.random_square_rotate(style, seed=seed)           
				style = np.expand_dims(style, axis=0)
				style = style[...,0:3]
				# print(style.shape)

				# TODO: Random augment the style
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
					order=3, 
					mode='reflect')

			else:
				# Adjust via callback 
				# Rotate 360 degree
				# Read the 3D image
				image = skimage.io.imread(images[rand_image])
				if image.shape != [DIMZ, DIMY, DIMX]: # Pad the image
					dimz, dimy, dimx = image.shape
					patz, paty, patx = (DIMZ-dimz)/2, (DIMY-dimy)/2, (DIMX-dimx)/2
					patz, paty, patx = int(patz), int(paty), int(patx)
					image = np.pad(image, ((patz, patz), (paty, paty), (patx, patx)), 
								   mode='constant', 
								   constant_values=0, 
						)
				image = np.transpose(image, [1, 2, 0])
				# image = np.expand_dims(image, axis=-1)
				image = np.expand_dims(image, axis=0)

				# Read the style
				style = skimage.io.imread(styles[rand_style])
				if style.ndim == 2: # If gray image, convert to 3 channel
					style = skimage.color.gray2rgb(style)
					# style = cv2.cvtColor(style, cv2.GRAY2RGB)
				seed = np.random.randint(0, 20152015)
				style = self.random_flip(style, seed=seed)        
				style = self.random_reverse(style, seed=seed)
				style = self.random_square_rotate(style, seed=seed)           
				style = np.expand_dims(style, axis=0)
				style = style[...,0:3]
				# print(style.shape)

				# TODO: Random augment the style
				# Resize if necessary 
				# style = skimage.transform.resize

				# Rotate and resample volume
				import scipy.ndimage.interpolation
				degrees = np.random.uniform(low=0.0, high=360.0)

				image = scipy.ndimage.interpolation.rotate(image, 
					angle=degrees, 
					axes=(0, 1), 
					reshape=False, #If reshape is true, the output shape is adapted so that the input 
								   #array is contained completely in the output. Default is True
					order=3, 
					mode='reflect')

			yield [image.astype(np.float32), 
				   style.astype(np.float32), 
				   ]
	def random_flip(self, image, seed=None):
		assert ((image.ndim == 2) | (image.ndim == 3))
		if seed:
			np.random.seed(seed)
		random_flip = np.random.randint(1,5)
		if random_flip==1:
			flipped = image[...,::1,::-1,:]
			image = flipped
		elif random_flip==2:
			flipped = image[...,::-1,::1,:]
			image = flipped
		elif random_flip==3:
			flipped = image[...,::-1,::-1,:]
			image = flipped
		elif random_flip==4:
			flipped = image
			image = flipped
		return image

	def random_reverse(self, image, seed=None):
		assert ((image.ndim == 2) | (image.ndim == 3))
		if seed:
			np.random.seed(seed)
		random_reverse = np.random.randint(1,3)
		if random_reverse==1:
			reverse = image[::1,...]
		elif random_reverse==2:
			reverse = image[::-1,...]
		image = reverse
		return image

	def random_square_rotate(self, image, seed=None):
		assert ((image.ndim == 2) | (image.ndim == 3))
		if seed:
			np.random.seed(seed)        
		random_rotatedeg = 90*np.random.randint(0,4)
		rotated = image.copy()
		from scipy.ndimage.interpolation import rotate
		if image.ndim==2:
			rotated = rotate(image, random_rotatedeg, axes=(0,1))
		elif image.ndim==3:
			rotated = rotate(image, random_rotatedeg, axes=(0,1)) # Channel
		image = rotated
		return image
				
	def random_elastic(self, image, seed=None):
		assert ((image.ndim == 2) | (image.ndim == 3))
		old_shape = image.shape

		if image.ndim==2:
			image = np.expand_dims(image, axis=0) # Make 3D
		new_shape = image.shape
		dimx, dimy = new_shape[1], new_shape[2]
		size = np.random.randint(4,16) #4,32
		ampl = np.random.randint(2, 5) #4,8
		du = np.random.uniform(-ampl, ampl, size=(size, size)).astype(np.float32)
		dv = np.random.uniform(-ampl, ampl, size=(size, size)).astype(np.float32)
		# Done distort at boundary
		du[ 0,:] = 0
		du[-1,:] = 0
		du[:, 0] = 0
		du[:,-1] = 0
		dv[ 0,:] = 0
		dv[-1,:] = 0
		dv[:, 0] = 0
		dv[:,-1] = 0
		import cv2
		from scipy.ndimage.interpolation    import map_coordinates
		# Interpolate du
		DU = cv2.resize(du, (new_shape[1], new_shape[2])) 
		DV = cv2.resize(dv, (new_shape[1], new_shape[2])) 
		X, Y = np.meshgrid(np.arange(new_shape[1]), np.arange(new_shape[2]))
		indices = np.reshape(Y+DV, (-1, 1)), np.reshape(X+DU, (-1, 1))
		
		warped = image.copy()
		for z in range(new_shape[0]): #Loop over the channel
			# print z
			imageZ = np.squeeze(image[z,...])
			flowZ  = map_coordinates(imageZ, indices, order=0).astype(np.float32)

			warpedZ = flowZ.reshape(image[z,...].shape)
			warped[z,...] = warpedZ     
		warped = np.reshape(warped, old_shape)
		return warped

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
def arch_generator(img, last_dim=3):
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
		dc = residual_dec('dc',    d0, DIMZ) 
		dd =  (LinearWrap(dc)
				.Conv2D('convlast', last_dim, kernel_shape=3, stride=1, padding='SAME', nl=tf.tanh, use_bias=True) ())
		return dd, dc

####################################################################################################
class Model(ModelDesc):
	def _get_inputs(self):
		return [
			InputDesc(tf.float32, (None, DIMY, DIMX, DIMZ), 'image'), # un comment line image = np.expand_dims
			# InputDesc(tf.float32, (DIMZ, DIMY, DIMX,    1), 'image'),
			InputDesc(tf.float32, (None, DIMY, DIMX,    3), 'style'),
			]
	#FusionNet
	@auto_reuse_variable_scope
	def generator(self, img, last_dim=3):
		assert img is not None
		return arch_generator(img, last_dim=last_dim)

	def _build_graph(self, inputs):
		G = tf.get_default_graph() # For round
		tf.local_variables_initializer()
		tf.global_variables_initializer()
		I, S = inputs # Get the image I and style S

		print(I)
		print(S)
		

		# Convert to range tanh
		I = tf_2tanh(I)
		S = tf_2tanh(S)

		with argscope([Conv2D, Deconv2D, FullyConnected],
					  W_init=tf.truncated_normal_initializer(stddev=0.02),
					  use_bias=False), \
				argscope(BatchNorm, gamma_init=tf.random_uniform_initializer()), \
				argscope([Conv2D, Deconv2D, BatchNorm], data_format='NHWC'), \
				argscope([Conv2D], dilation_rate=1):

			R, V = self.generator(I, last_dim=3) # Generate the rendering from image I


		# Calculating loss goes here
		def additional_losses(a, b):
			VGG_MEAN = np.array([123.68, 116.779, 103.939])  # RGB
			VGG_MEAN_TENSOR = tf.constant(VGG_MEAN, dtype=tf.float32)

			def normalize(v):
				assert isinstance(v, tf.Tensor)
				v.get_shape().assert_has_rank(4)
				return v / tf.reduce_mean(v, axis=[1, 2, 3], keepdims=True)


			def gram_matrix(v):
				assert isinstance(v, tf.Tensor)
				v.get_shape().assert_has_rank(4)
				dim = v.get_shape().as_list()
				v = tf.reshape(v, [-1, dim[1] * dim[2], dim[3]])
				return tf.matmul(v, v, transpose_a=True)
	
			with tf.variable_scope('VGG19'):
				x = tf.concat([a, b], axis=0)
				#x = tf.reshape(x, [2 * BATCH_SIZE, SHAPE_LR * 4, SHAPE_LR * 4, 3]) * 255.0
				x = tf_2imag(x) # convert to range image
				x = x - VGG_MEAN_TENSOR
				# VGG 19
				with varreplace.freeze_variables():
					with argscope(Conv2D, kernel_shape=3, nl=tf.nn.relu):
						conv1_1 = Conv2D('conv1_1', x, 64)
						conv1_2 = Conv2D('conv1_2', conv1_1, 64)
						pool1 = MaxPooling('pool1', conv1_2, 2)  # 64
						conv2_1 = Conv2D('conv2_1', pool1, 128)
						conv2_2 = Conv2D('conv2_2', conv2_1, 128)
						pool2 = MaxPooling('pool2', conv2_2, 2)  # 32
						conv3_1 = Conv2D('conv3_1', pool2, 256)
						conv3_2 = Conv2D('conv3_2', conv3_1, 256)
						conv3_3 = Conv2D('conv3_3', conv3_2, 256)
						conv3_4 = Conv2D('conv3_4', conv3_3, 256)
						pool3 = MaxPooling('pool3', conv3_4, 2)  # 16
						conv4_1 = Conv2D('conv4_1', pool3, 512)
						conv4_2 = Conv2D('conv4_2', conv4_1, 512)
						conv4_3 = Conv2D('conv4_3', conv4_2, 512)
						conv4_4 = Conv2D('conv4_4', conv4_3, 512)
						pool4 = MaxPooling('pool4', conv4_4, 2)  # 8
						conv5_1 = Conv2D('conv5_1', pool4, 512)
						conv5_2 = Conv2D('conv5_2', conv5_1, 512)
						conv5_3 = Conv2D('conv5_3', conv5_2, 512)
						conv5_4 = Conv2D('conv5_4', conv5_3, 512)
						pool5 = MaxPooling('pool5', conv5_4, 2)  # 4

			# perceptual loss
			with tf.name_scope('perceptual_loss'):
				pool2 = normalize(pool2)
				pool5 = normalize(pool5)
				phi_a_1, phi_b_1 = tf.split(pool2, 2, axis=0)
				phi_a_2, phi_b_2 = tf.split(pool5, 2, axis=0)

				logger.info('Create perceptual loss for layer {} with shape {}'.format(pool2.name, pool2.get_shape()))
				pool2_loss = tf.losses.mean_squared_error(phi_a_1, phi_b_1, reduction=tf.losses.Reduction.MEAN)
				logger.info('Create perceptual loss for layer {} with shape {}'.format(pool5.name, pool5.get_shape()))
				pool5_loss = tf.losses.mean_squared_error(phi_a_2, phi_b_2, reduction=tf.losses.Reduction.MEAN)

			# texture loss
			with tf.name_scope('texture_loss'):
				def texture_loss(x, p=16):
					_, h, w, c = x.get_shape().as_list()
					x = normalize(x)
					assert h % p == 0 and w % p == 0
					logger.info('Create texture loss for layer {} with shape {}'.format(x.name, x.get_shape()))

					x = tf.space_to_batch_nd(x, [p, p], [[0, 0], [0, 0]])  # [b * ?, h/p, w/p, c]
					x = tf.reshape(x, [p, p, -1, h // p, w // p, c])       # [p, p, b, h/p, w/p, c]
					x = tf.transpose(x, [2, 3, 4, 0, 1, 5])                # [b * ?, p, p, c]
					patches_a, patches_b = tf.split(x, 2, axis=0)          # each is b,h/p,w/p,p,p,c

					patches_a = tf.reshape(patches_a, [-1, p, p, c])       # [b * ?, p, p, c]
					patches_b = tf.reshape(patches_b, [-1, p, p, c])       # [b * ?, p, p, c]
					return tf.losses.mean_squared_error(
						gram_matrix(patches_a),
						gram_matrix(patches_b),
						reduction=tf.losses.Reduction.MEAN
					)

				texture_loss_conv1_1 = tf.identity(texture_loss(conv1_1), name='normalized_conv1_1')
				texture_loss_conv2_1 = tf.identity(texture_loss(conv2_1), name='normalized_conv2_1')
				texture_loss_conv3_1 = tf.identity(texture_loss(conv3_1), name='normalized_conv3_1')

			return [pool2_loss, pool5_loss, texture_loss_conv1_1, texture_loss_conv2_1, texture_loss_conv3_1]

		additional_losses = additional_losses(R, S) # Concat Rendering and Style
		with tf.name_scope('additional_losses'):
			# see table 2 from appendix
			loss = []
			#loss.append(tf.multiply(GAN_FACTOR_PARAMETER, self.g_loss, name="loss_LA"))
			loss.append(tf.multiply(2e-1, additional_losses[0], name="loss_LP1"))
			loss.append(tf.multiply(2e-2, additional_losses[1], name="loss_LP2"))
			loss.append(tf.multiply(3e-7, additional_losses[2], name="loss_LT1"))
			loss.append(tf.multiply(1e-6, additional_losses[3], name="loss_LT2"))
			loss.append(tf.multiply(1e-6, additional_losses[4], name="loss_LT3"))

			loss.append(tf.reduce_mean(tf.abs(V-I), name="loss_abs"))

		if get_current_tower_context().is_training:
			self.cost = tf.add_n(loss, name='cost')
			add_moving_summary(self.cost)

		# Visualization
		def visualize(x, name='viz'):
			viz = tf_2imag(x)
			viz = tf.cast(tf.clip_by_value(viz, 0, 255), tf.uint8, name=name)
			tf.summary.image(name, viz, max_outputs=30) #max(30, BATCH_SIZE)

		visualize(tf.transpose(I[...,128-5:128+5,:,:], [1, 2, 3, 0]), name='viz_image')
		visualize(S, name='viz_style')
		visualize(R, name='rendering')

	def _get_optimizer(self):
		lr  = tf.get_variable('learning_rate', initializer=1e-4, trainable=False)
		opt = tf.train.AdamOptimizer(lr)
		return opt

###################################################################################################
class VisualizeRunner(Callback):
	def _setup_graph(self):
		self.pred = self.trainer.get_predictor(
			['image', 'style'], ['rendering'])

	def _before_train(self):
		global args
		self.ds_train, self.ds_valid = get_data(args.image, args.style)

	def _trigger(self):
		for lst in self.ds_valid.get_data():
			viz_valid = self.pred(lst)
			viz_valid = np.squeeze(np.array(viz_valid))

			#print viz_valid.shape

			self.trainer.monitors.put_image('viz_valid', viz_valid)
###################################################################################################
def apply(model_path, image_path, style_path):
	pass

###################################################################################################
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', 	help='comma separated list of GPU(s) to use.')
	parser.add_argument('--load', 	help='load model')
	parser.add_argument('--apply', 	action='store_true')
	parser.add_argument('--image', 	help='path to the image. ', default="data/image/")
	parser.add_argument('--style',  help='path to the style. ', default="data/style/zebra_1")
	parser.add_argument('--vgg19', 	help='load model', 			default="data/vgg19.npz")
	parser.add_argument('--output', help='directory for saving the rendering', default=".", type=str)
	args = parser.parse_args()
	print(args)
	parser.print_help()


	if args.gpu:
		os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

	if args.apply:
		apply(args.load, args.image, args.style)
	else:
		# Set the logger directory
		logger.auto_set_dir()

		nr_tower = max(get_nr_gpu(), 1)
		# ds_train, ds_valid = QueueInput(get_data(args.image, args.style))
		ds_train, ds_valid = get_data(args.image, args.style)

		ds_train = PrintData(ds_train)
		ds_valid = PrintData(ds_valid)

		ds_train = PrefetchDataZMQ(ds_train, 8)
		ds_valid = PrefetchDataZMQ(ds_valid, 1)
		


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
					# PeriodicTrigger(VisualizeRunner(), every_k_epochs=5),
					# PeriodicTrigger(InferenceRunner(ds_valid, [ScalarStats('loss_membr')]), every_k_epochs=5),
					ScheduledHyperParamSetter('learning_rate', [(0, 2e-4), (100, 1e-4), (200, 1e-5), (300, 1e-6)], interp='linear')
					# ScheduledHyperParamSetter('learning_rate', [(30, 6e-6), (45, 1e-6), (60, 8e-7)]),
					# HumanHyperParamSetter('learning_rate'),
					],
				max_epoch       =   500, 
				session_init    =   session_init,
				nr_tower        =   max(get_nr_gpu(), 1)
				)
		
			# Train the model
			SyncMultiGPUTrainer(config).train()

	
	
		