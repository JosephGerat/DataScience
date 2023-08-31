import math
from typing import List
num_friends = [100.0,49,41,40,25,21,21,19,19,18,18,16,15,15,15,15,14,14,13,13,13,13,12,12,11,10,10,10,10,10,10,10,10,
               10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,
               7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,
               4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
               1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]


def mean(xs: List[float]):
    return sum(xs) / len(xs)


assert mean([0, 1, 2]) == 1.0

def variance(xs: List[float]):
    n = len(xs)

    m = mean(xs)
    s = 0
    for i in xs:
        s += pow(i - m, 2)

    return s/(n-1)

def standard_deviation(xs: List[float]):
    return math.sqrt(variance(xs))

def quantile(xs: List[float], p: float):
    """ Returns pth-percentile value in x"""
    sorted(xs)

v = variance(num_friends)
assert 81.54 < v < 81.55

sd = standard_deviation(num_friends)
print(sd)
