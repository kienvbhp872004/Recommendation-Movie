import os
from os.path import abspath

def getpath() :
    path = abspath(__file__).replace("path.py", "")
    return path
