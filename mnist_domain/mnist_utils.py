import itertools 
import numpy as np
import random
from constants import *


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

# Compute the quantile value corresponding to the point-predictor's output
def get_w_alg():
    # imgs = load_mnist() # Assuming this dataset to be output of alg
    scores = list()
    for img in MNIST_IMGS:
        label = img.gt
        score = img.preds[label]
        scores.append(-1 * score) # a lower score indicates more confidence about the prediction

    return get_w_conformal(scores)

# Most relevant function: Given a list of non-conformity scores, return the appropriate threshold value corresponding to (1 - delta) coverage
def get_w_conformal(scores):
     calibration_size = len(scores)
     desired_quantile = np.ceil((1 - MNIST_DELTA) * (calibration_size + 1)) / calibration_size
     chosen_quantile = np.minimum(1.0, desired_quantile)
     w = np.quantile(scores, chosen_quantile)
     return w

MNIST_IMGS = load_mnist()
PRED_SET_THRESHOLD = get_w_alg()

def get_conf(cur_int):
    ls = itertools.product(*[get_pred_set(digit_img, PRED_SET_THRESHOLD) for digit_img in cur_int])
    return [sum([item * 10**i for (i, item) in enumerate(l)]) for l in ls]

def get_gt(img_list):
    return sum([img.gt * 10**i for (i, img) in enumerate(img_list)])

def get_standard(img_list):
    return sum([img.get_pred() * 10**i for (i, img) in enumerate(img_list)])

# Returns set-valued prediction over {0, 1, ..., 9}
def get_pred_set(img, w):
    pred_set = []
    for i, val in enumerate(img.preds):
        if (-1 * val) <= w:
            pred_set.append(i)
    return pred_set

def get_int(correct_preds, wrong_preds):
    digit_list = []
    for _ in range(DIGITS_PER_ITEM):
        if random.random() > MNIST_NOISE:
            digit_list.append(random.choice(correct_preds))
        else:
            digit_list.append(random.choice(wrong_preds))
    return digit_list
    

def join(pred1, pred2):
    return (min(pred1[0], pred2[0]), max(pred1[1], pred2[1]))


# TODO: DELETE if not necessary
# def join_intervals(interval1, interval2):

#     new_interval = (max(interval1[0], interval2[0]), min(interval1[1], interval2[1]))
#     if new_interval[0] > new_interval[1]:
#         return None 
#     return new_interval

def intersect_intervals(interval1, interval2):

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