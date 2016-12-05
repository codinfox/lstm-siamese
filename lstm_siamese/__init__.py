from __future__ import division, print_function, absolute_import
import os.path

result_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
print("result root is at {}".format(result_root))
dir_dictionary = {
    'datasets': os.path.join(result_root, 'datasets'),
    'features': os.path.join(result_root, 'features'),
    'models': os.path.join(result_root, 'models'),
    'root': os.path.join(result_root, '..'),
}