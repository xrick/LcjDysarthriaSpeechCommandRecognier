"""""""""""""""
This file contains operations related to file system
"""""""""""""""
import os

def getNextLevelDirs(root):
    return next(os.walk(root))[1]
