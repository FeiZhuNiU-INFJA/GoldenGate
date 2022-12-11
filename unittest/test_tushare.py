import sys
import unittest
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))
from data.tushare_api import *


class BasicFunctions(unittest.TestCase):

    def test_get_symbol(self):
        data = get_symbols()
        print(data.head())
        self.assertTrue(len(data) > 100)


if __name__ == '__main__':
    unittest.main()