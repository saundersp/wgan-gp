import numpy as np
from numpy.typing import NDArray
from math import floor
from types import ModuleType
from typing import Final, Optional

def extract_mnist(data: ModuleType, img_shape: tuple[int, int, int] = (28, 28, 1),
				  label: Optional[NDArray[np.uint8]] = None) -> NDArray[np.float32]:
	(X_train, y_train), (x_test, y_test) = data.load_data()

	if label:
		X_train = X_train[(y_train.reshape(-1) == label)]
		x_test = x_test[(y_test.reshape(-1) == label)]

	X: NDArray[np.float32] = np.concatenate([X_train, x_test])

	# Reshaping
	X = X.reshape((
		X.shape[0],
		img_shape[0],
		img_shape[1],
		img_shape[2])).astype(np.float32)
	# Normalization to [-1, 1]
	X = X / 127.5 - 1.0

	return X

formats: Final[list[str]] = ['s', 'm', 'h', 'j', 'w', 'M', 'y']
nb: Final[NDArray[np.uint8]] = np.asarray([1, 60, 60, 24, 7, 4, 12], dtype = np.uint8)
def format_time(time: float) -> str:
	prod: int = nb.prod()

	s: str = ''
	for i in range(nb.shape[0])[::-1]:
		if time >= prod:
			res: float = floor(time / prod)
			time %= prod
			s += f'{res}{formats[i]} '
		prod /= nb[i]
	return s.rstrip()
