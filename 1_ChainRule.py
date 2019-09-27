import numpy as np
from numpy import array as ndarray;
from typing import Callable, List

def square(x: ndarray) -> ndarray:
	'''
	square each element in the input array
	'''
	return np.power(x, 2)

def sigmoid(x: np.array) -> np.array:
	'''
	apply the sigmoid function to each element in the input array
	'''
	return 1 / (1 + np.exp(-x))

def deriv(func: Callable[[ndarray], ndarray],
		  input_: ndarray,
		  delta: float = 0.001) -> ndarray:
	'''
	evaluates the derivative of a functon "func" at every element
	in the "input_" array
	'''
	return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)

# a function takes in an array as an argument and produces an array
Array_Function = Callable[[ndarray], ndarray]

# a chain is a list of functions
Chain = List[Array_Function]

def chain_deriv_2(chain: Chain, input_range: np.array) -> np.array:
	'''
	uses the chain rule to compute the derivative of two nested funcs
	(f2(f1(x))' = f2'(f1(x)) * f1'(x)
	'''
	assert len(chain) == 2, \
	"This function requires 'Chain' objects of length 2"
	assert input_range.ndim == 1, \
	"Function requires a 1 dimensional array as input_range"
	
	f1 = chain[0]
	f2 = chain[1]
	
	# df1/dx
	f1_of_x = f1(input_range)
	# df1/du
	df1dx = deriv(f1, input_range)
	# df2/du(f1(x))
	df2du = deriv(f2, f1(input_range))
	# Multiplying these quantities together at each point
	return df1dx * df2du
	
INPUT_RANGE = np.arange(-3, 3, 0.01)
chain_1 = [square, sigmoid]
chain_2 = [sigmoid, square]

print(chain_deriv_2(chain_1, INPUT_RANGE))
print(chain_deriv_2(chain_2, INPUT_RANGE))
