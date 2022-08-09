import unittest
import os
import sys

sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','src'))

class TestEnvironment(unittest.TestCase):
	"""
	TODO get rid of this testcase
	"""
	def test_imports(self):
		from elcc_impl import main
		pass

if __name__ == "__main__":
	unittest.main()