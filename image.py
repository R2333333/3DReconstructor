import numpy as np
import sys
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from mapping import mapping
import cv2

def drawSame():
    front, left, top = np.zeros((200,200)), np.zeros((200,200)), np.zeros((200,200))
    left[50:150, 50:150] = (left + 1)[50:150, 50:150]
    top[50:150, 25:175] = (top + 1)[50:150, 25:175]
    front[50:150, 25:175] = (front + 1)[50:150, 25:175]

    return front, left, top

def drawDiff():
    front, left, top = np.zeros((200,200)), np.zeros((200,200)), np.zeros((200,200))
    left[50:150, 50:150] = (left + 1)[50:150, 50:150]
    top[50:150, 25:175] = (top + 1)[50:150, 25:175]
    front[50:150, 25:175] = (front + 1)[50:150, 25:175]

    return front, left, top

