# get used to type-checked function
import numpy as np

# add type to func args and returns
def square(x: np.array) -> np.array:
	# add proper doc to each function
	'''
	square each element in the input np.array
	'''
	return np.power(x, 2)
	
def leaky_relu(x: np.array) -> np.array:
	'''
	apply "Leaky ReLU" func to each element in the input np.array
	'''
	return np.maximum(0.2 * x, x)

a = np.array([[1,2,3],[4,5,6]])
print("a:\n",a)
print("square(a):\n",square(a))
print("leaky_relu(a):\n",leaky_relu(a))
