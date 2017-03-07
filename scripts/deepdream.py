#!/usr/bin/env python2

# pylint: disable=E0401
import numpy as np
import scipy.ndimage as nd
from PIL import Image
from IPython.display import clear_output
from google.protobuf import text_format

import caffe
# pylint: enable=E0401

from argparse import ArgumentParser

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(classifier, image):
	return np.float32(np.rollaxis(image, 2)[::-1]) - classifier.transformer.mean['data']

def deprocess(classifier, image):
	return np.dstack((image + classifier.transformer.mean['data'])[::-1])

def objective_L2(dst):
	dst.diff[:] = dst.data

def make_step(classifier, step_size=1.5, end='inception_4c/output',
			  jitter=32, clip=True, objective=objective_L2):
	'''Basic gradient ascent step.'''

	src = classifier.blobs['data'] # input image is stored in Net's 'data' blob
	dst = classifier.blobs[end]

	ox, oy = np.random.randint(-jitter, jitter+1, 2)
	src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift

	classifier.forward(end=end)
	objective(dst)  # specify the optimization objective
	classifier.backward(start=end)
	g = src.diff[0]
	# apply normalized ascent step to the input image
	src.data[:] += step_size/np.abs(g).mean() * g

	src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image

	if clip:
		bias = classifier.transformer.mean['data']
		src.data[:] = np.clip(src.data, -bias, 255 - bias)

def deepdream(classifier, base_img, iterations=10, num_octaves=4, octave_scale=1.4,
			  end='inception_4c/output', clip=True, **step_params):
	# prepare base images for all octaves
	octaves = [preprocess(classifier, base_img)]
	for _ in range(num_octaves - 1):
		octaves.append(nd.zoom(octaves[-1], (1, 1.0 / octave_scale, 1.0 / octave_scale), order=1))

	src = classifier.blobs['data']
	detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
	for octave, octave_base in enumerate(octaves[::-1]):
		h, w = octave_base.shape[-2:]
		if octave > 0:
			# upscale details from the previous octave
			h1, w1 = detail.shape[-2:]
			detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

		src.reshape(1,3,h,w) # resize the network's input image size
		src.data[0] = octave_base+detail
		for i in range(iterations):
			make_step(classifier, end=end, clip=clip, **step_params)

			# visualization
			vis = deprocess(classifier, src.data[0])
			if not clip: # adjust image contrast if clipping is disabled
				vis = vis*(255.0/np.percentile(vis, 99.98))

			print(octave, i, end, vis.shape)
			clear_output(wait=True)

		# extract details produced on the current octave
		detail = src.data[0]-octave_base
	# returning the resulting image
	return deprocess(classifier, src.data[0])

def save_image(image, outfile):
	image = np.uint8(np.clip(image, 0, 255))
	Image.fromarray(image).save(outfile)

def get_classifier(model_name):
	# If your GPU supports CUDA and Caffe was built with CUDA support,
	# uncomment the following to run Caffe operations on the GPU.
	# caffe.set_mode_gpu()
	# caffe.set_device(0) # select GPU device if multiple devices exist

	model_path = '/opt/caffe/models/{}/'.format(model_name) # substitute your path here
	net_fn = model_path + 'deploy.prototxt'
	param_fn = model_path + '{}.caffemodel'.format(model_name)

	# Patching model to be able to compute gradients.
	# Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
	model = caffe.io.caffe_pb2.NetParameter() # pylint: disable=E1101
	text_format.Merge(open(net_fn).read(), model)
	model.force_backward = True
	open('tmp.prototxt', 'w').write(str(model))


	return caffe.Classifier('tmp.prototxt', param_fn, # pylint: disable=E1101
						   mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
						   channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

def main():
	parser = ArgumentParser(description='Run deepdream')
	parser.add_argument('input', action='store', help='Path to input file')
	parser.add_argument('output', action='store', help='Path to output file')
	parser.add_argument('-m', '--model', dest='model', default='bvlc_googlenet')
	parser.add_argument('-i', '--iterations', dest='iterations', default=10)
	parser.add_argument('-o', '--octaves', dest='octaves', default=4)
	parser.add_argument('-s', '--scale', dest='scale', default=1.4)
	parser.add_argument('-e', '--end', dest='end', default='inception_4c/output')
	parser.add_argument('-g', '--guide', dest='guide')

	args = parser.parse_args()

	classifier = get_classifier(args.model)

	input_image = np.float32(Image.open(args.input))

	out = deepdream(classifier, input_image, iterations=args.iterations,
					num_octaves=args.octaves, octave_scale=args.scale,
					end=args.end)

	save_image(out, args.output)

if __name__ == '__main__':
	main()
