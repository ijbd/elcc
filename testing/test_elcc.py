import unittest
import os
import sys

sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','src'))

class TestGeneratorMethods(unittest.TestCase):
	"""
	TODO This shouldn't be in the test_elcc file, just as this function shouldn't be in the elcc_impl file
	"""
	def test_find_nearest_impl(self):
		from elcc_impl import find_nearest_impl
		discrete_coordinates = [0,1]
		test_coordinates = [-1, .4, .6, 1.5]
		test_indices = find_nearest_impl(test_coordinates, discrete_coordinates)
		expected_indices = [0,0,1,1]
		self.assertEqual(test_indices, expected_indices)

if __name__ == "__main__":
	unittest.main()