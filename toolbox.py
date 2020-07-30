import numpy as np
import math

def extract_mnist(data, img_shape = (28, 28, 1), label = None):
	(x_train, y_train), (x_test, y_test) = data.load_data()

	if label:
		x_train = x_train[(y_train.reshape(-1) == label)]
		x_test = x_test[(y_test.reshape(-1) == label)]

	x_train = np.concatenate([x_train, x_test])

	# Reshaping
	x_train = x_train.reshape((
		x_train.shape[0],
		img_shape[0],
		img_shape[1],
		img_shape[2])).astype(np.float32)
	# Normalization to [-1, 1]
	x_train = x_train / 127.5 - 1.0

	return x_train

def format_time(time):
	formats = ["s", "m", "h", "j", "w", "M", "y"]
	nb = np.array([1, 60, 60, 24, 7, 4, 12])
	prod = nb.prod()

	s = ""
	for i in range(nb.shape[0])[::-1]:
		if time >= prod:
			res = math.floor(time / prod)
			time %= prod
			s += f"{res}{formats[i]} "
		prod /= nb[i]
	return s.rstrip()