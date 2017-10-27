#! /usr/bin/env python2

import sys

last_x = 0.5
last_y = 0.5

x_changed = False
y_changed = False

for line in open(sys.argv[1]):
    if line.startswith('/'):
        line = line[8:]
    parts = line.split(" ")
    if parts[0] == "Y":
        new_y = float(parts[2])
        if new_y != last_y:
            last_y = new_y
            y_changed = True
    else:
        new_x = float(parts[2])
        if new_x != last_x:
            last_x = new_x
            x_changed = True
    if x_changed and y_changed:
        x_changed = False
        y_changed = False
        print last_x, last_y
