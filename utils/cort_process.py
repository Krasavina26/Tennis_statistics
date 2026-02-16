import numpy as np
from sympy import Line 
import sympy

def line_intersection(line1, line2):
    """
    Find 2 lines intersection point
    """
    l1 = Line((line1[0], line1[1]), (line1[2], line1[3]))
    l2 = Line((line2[0], line2[1]), (line2[2], line2[3]))

    intersection = l1.intersection(l2)
    point = None
    if len(intersection) > 0:
        if isinstance(intersection[0], sympy.geometry.point.Point2D):
            point = intersection[0].coordinates
    return point

def make_parallel_line(line1, net_y, width_frame):
    y1 = line1[1]
    y2 = line1[3]

    dy = abs(y1 - y2)

    if y1 < y2:
        return [0, net_y-dy, width_frame, net_y]

    else:
        return [0, net_y, width_frame, net_y-dy]


def is_point_in_image(x, y, input_width=1280, input_height=720):
    res = False
    if x and y:
        res = (x >= 0) and (x <= input_width) and (y >= 0) and (y <= input_height)
    return res