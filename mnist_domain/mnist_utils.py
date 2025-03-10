import itertools 
import numpy as np
import random
from constants import *
import json
import math


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class Image:
    def __init__(self, preds, gt):
        self.preds = preds
        self.gt = gt

    def get_pred(self):
        return np.argmax(self.preds)

    def __str__(self):
        return "({}, {})".format(self.get_pred(), self.gt)

def load_mnist():
    imgs = []
    with open(MNIST_IMGS_DIR) as f:
        for line in f:
            toks = line.strip().split(",")
            preds = np.array([float(x) for x in toks[:10]])
            gt = int(toks[10][0])
            imgs.append(Image(preds, gt))
    return imgs

def get_w_alg():
    '''
    We use a dataset of labeled MNIST digits to compute a thresholding score. This score ensures that 
    1) The ground truth value will be contained in the prediction set with 1 - \delta probability.
    2) The prediction sets will be as small as possible while still guaranteeing (1)
    '''
    scores = list()
    for img in MNIST_IMGS:
        label = img.gt
        score = img.preds[label]
        scores.append(-1 * score) # a lower score indicates more confidence about the prediction

    return get_w_conformal(scores)


def get_w_conformal(scores):
     '''
     A helper function for computing the thresholding score. Given a list of non-conformity scores from the calibration dataset,
     return the appropriate thresholding score corresponding to (1 - delta) coverage. 
     '''
     calibration_size = len(scores)
     desired_quantile = np.ceil((1 - MNIST_DELTA) * (calibration_size + 1)) / calibration_size
     chosen_quantile = np.minimum(1.0, desired_quantile)
     w = np.quantile(scores, chosen_quantile)
     return w

MNIST_IMGS = load_mnist()
PRED_SET_THRESHOLD = get_w_alg()

def get_conf(cur_int):
    ls = itertools.product(*[get_pred_set(digit_img, PRED_SET_THRESHOLD) for digit_img in cur_int])
    return [int(sum([item * 10**i for (i, item) in enumerate(l)])) for l in ls]

def get_probs(cur_int):
    ls = itertools.product(*[get_pred_set_probs(digit_img, PRED_SET_THRESHOLD) for digit_img in cur_int])
    # return list(ls)
    return [math.prod(l) for l in ls]


def get_gt(img_list):
    return sum([img.gt * 10**i for (i, img) in enumerate(img_list)])

def get_standard(img_list):
    return sum([img.get_pred() * 10**i for (i, img) in enumerate(img_list)])

def get_pred_set(img, w):
    '''
    Computes the prediction set of a given MNIST digit w.r.t. a thresholding score w.
    '''
    pred_set = []
    for i, val in enumerate(img.preds):
        if (-1 * val) <= w:
            pred_set.append(i)
    return pred_set

def get_pred_set_probs(img, w):
    '''
    Computes the prediction set of a given MNIST digit w.r.t. a thresholding score w.
    '''
    probs = []
    for i, val in enumerate(img.preds):
        if (-1 * val) <= w:
            probs.append(val)
    return probs


def get_int(imgs):
    '''
    Creates an integer comprised of a predetermined number of MNIST digits.
    '''
    digit_list = []
    for _ in range(DIGITS_PER_ITEM):
        digit_list.append(random.choice(imgs))
    return digit_list

def get_new_img_list(inp, imgs):
    img_list = []
    for i in range(3):
        possible_imgs = [img for img in imgs if get_gt([img]) == inp['gt']["img-list"][i] and get_standard([img]) == inp['standard']["img-list"][i] and get_conf([img]) == inp['conf']["img-list"][i]]
        img_list.append(random.choice(possible_imgs))
    possible_imgs = [img for img in imgs if get_gt([img]) == inp['gt']["img"] and get_standard([img]) == inp['standard']["img"] and get_conf([img]) == inp['conf']["img"]]
    img = random.choice(possible_imgs)
    return img_list, img
    

def join(pred1, pred2):
    return (min(pred1[0], pred2[0]), max(pred1[1], pred2[1]))


def intersect_intervals(interval1, interval2):
    '''
    Compute the intersection of 2 integer intervals in the MNIST domain.
    '''
    if interval2 is None:
        raise TypeError
        return None

    start1, end1 = interval1
    start2, end2 = interval2
    
    # Check if intervals overlap
    if end1 < start2 or end2 < start1:
        return None  # No intersection
    
    # Calculate intersection
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    
    return (intersection_start, intersection_end)