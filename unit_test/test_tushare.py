import sys
import unit_test
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))
from data.tushare_api import *


class BasicFunctions(unit_test.TestCase):

    def test_get_symbol(self):
        data = get_symbols()
        print(data.head())
        self.assertTrue(len(data) > 100)


if __name__ == '__main__':
    unit_test.main()