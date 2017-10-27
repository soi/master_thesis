#! /usr/bin/env python
from __future__ import print_function, division
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

log = sys.argv[1]

count = 0
loss = []
val_loss = []
for line in open(log):
    if count > 0:
        parts = line.split(',')
        if len(parts) > 2:
            loss.append(float(parts[1]))
            val_loss.append(float(parts[2]))
    count += 1

ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.plot(loss, label='Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.show()
