
import math

for i in [0.01, 0.1, 0.5, 1, 2]:
    o = math.log(i)
    print(f'log({i}) -> {o}')

    exp_o = math.exp(o)
    print(f'exp({exp_o})')