'''
some assistant functions
'''
import os

def ensure_dir(d):
    if not os.path.exists(d):
        print("Directory {} do not exist; creating...".format(d))
        os.makedirs(d)
