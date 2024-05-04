import math
def point_between(a, b, r):

    d = math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
    if(d<r):
        return b[0], b[1]
    cx = a[0] + r * (b[0] - a[0]) / d
    cy = a[1] + r * (b[1] - a[1]) / d

    return cx, cy