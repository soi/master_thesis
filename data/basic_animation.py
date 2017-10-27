#! /usr/bin/env pytho)
from __future__ import print_function, division
try:
    import tkinter as tk
except ImportError:
    import Tkinter as tk # python2
import numpy as np
import random
import matplotlib.pyplot as plt
import time
from terminaltables import AsciiTable
from subprocess import call

class Animation(tk.Frame):
    WINDOW_SIZE_Y = 700
    WINDOW_SIZE_X = int(WINDOW_SIZE_Y / 1.777777)
    PIXEL_SIZE = 6
    CONFIG = {
            'bg_color': '#ffffff',
            'step_delay': 40,
            'step_offset': 10,
            'input': {
                'visible': True,
                'color': 'blue'
            }}


    def __init__(self, input_paths):
        self.input_paths = input_paths
        self.input_len = len(input_paths[0])

        self.master = tk.Tk()
        self.full_rect, self.canvas = self.init_canvas(self.CONFIG['bg_color'])
        self.input_rects = self.init_rects(self.input_len,
                                          self.CONFIG['input']['color'],
                                          fade_to_last=False)

        # GUI elements
        self.button_frame = tk.Frame(self.master)

        self.go_to_start_button = tk.Button(self.button_frame, text='Go to start')
        self.go_to_start_button.pack(side='left')
        self.go_to_start_button.bind('<ButtonPress-1>', self.go_to_start)

        self.back_button = tk.Button(self.button_frame, text='<')
        self.back_button.bind('<ButtonPress-1>', self.go_back)
        self.back_button.pack(side='left')

        self.pause_button = tk.Button(self.button_frame, text='Pause')
        self.pause_button.pack(side='left')
        self.pause_button.bind('<ButtonPress-1>', self.pause)

        self.forward_button = tk.Button(self.button_frame, text='>')
        self.forward_button.pack(side='left')
        self.forward_button.bind('<ButtonPress-1>', self.go_forward)
        self.button_frame.pack(side='bottom')

        self.faster_button = tk.Button(self.button_frame, text='Faster')
        self.faster_button.bind('<ButtonPress-1>', self.faster)
        self.faster_button.pack(side='left')

        self.slower_button = tk.Button(self.button_frame, text='Slower')
        self.slower_button.bind('<ButtonPress-1>', self.slower)
        self.slower_button.pack(side='left')

        self.save_button = tk.Button(self.button_frame, text='save')
        self.save_button.bind('<ButtonPress-1>', self.save)
        self.save_button.pack(side='left')

        self.text = tk.Label(self.master, text=0)
        self.text.pack(side='bottom')

        self.delay_text = tk.Label(self.master, text=self.CONFIG['step_delay'],
                fg='black')
        self.delay_text.pack(side='right')

        # start from the train/test boundary
        self.count = 0
        self.pause = False
        self.update()
        self.master.mainloop()

    def init_canvas(self, bg_color):
        canvas = tk.Canvas(self.master, height=self.WINDOW_SIZE_Y,
                width=self.WINDOW_SIZE_X)
        full_rect = canvas.create_rectangle(0, 0, self.WINDOW_SIZE_X,
                self.WINDOW_SIZE_Y, outline='', fill=bg_color)
        canvas.pack()
        return full_rect, canvas

    def init_rects(self, count, color, fade_to_last=False):
        rects = []
        for _ in range(count):
            rect = self.canvas.create_rectangle(0, 0, 1, 1, outline='')
            rects.append(rect)
        self.change_rect_color(rects, color, fade_to_last=fade_to_last)
        return rects

    def change_rect_color(self, rects, color, fade_to_last=False):
        # only fade to half the opacity
        if fade_to_last:
            offset = 128 / len(rects)
            color_int = 0
        else:
            offset = -128 / len(rects)
            color_int = 128

        for i, rect in enumerate(rects):
            color_str_part = str(hex(int(color_int)))[2:].zfill(2)
            if color is 'red':
                color_str = '#ff' + 2 * color_str_part
                self.canvas.itemconfig(rect, fill=color_str)
            if color is 'green':
                color_str = '#' + color_str_part + 'ff' + color_str_part
                self.canvas.itemconfig(rect, fill=color_str)
            if color is 'blue':
                color_str = '#' + color_str_part * 2 + 'ff'
                self.canvas.itemconfig(rect, fill=color_str)
            color_int += offset

    def go_to_start(self, event):
        print('start')
        self.pause = True
        self.count = 0
        self.update()

    def go_back(self, event):
        self.pause = True
        self.count -= 1
        self.update()

    def go_forward(self, event):
        self.pause = True
        self.count += 1
        self.update()

    def pause(self, event):
        print('pause')
        self.pause = not self.pause
        if not self.pause:
            self.pause_button.config(text='Pause')
            self.update()
        else:
            self.pause_button.config(text='Play')

    def faster(self, event):
        new_delay = self.CONFIG['step_delay'] - self.CONFIG['step_offset']
        self.CONFIG['step_delay'] = max(1, new_delay)
        self.delay_text.config(text=self.CONFIG['step_delay'])

    def slower(self, event):
        self.CONFIG['step_delay'] += self.CONFIG['step_offset']
        self.delay_text.config(text=self.CONFIG['step_delay'])

    def save(self, event):
        unix_time = time.time()
        self.canvas.postscript(file="saved_image" + str(unix_time) + ".ps",
                               colormode='color')
        print("saved image")

    def get_coord(self, data_point):
        max_size_x = self.WINDOW_SIZE_X / self.PIXEL_SIZE
        max_size_y = self.WINDOW_SIZE_Y / self.PIXEL_SIZE
        x = int(data_point[0] * max_size_x) * self.PIXEL_SIZE
        y = int(data_point[1] * max_size_y) * self.PIXEL_SIZE
        return x, y

    def set_rect_coords(self, data, rects):
        for i, data_point in enumerate(data):
            point = self.get_coord(data_point)
            self.canvas.coords(rects[i], point[0], point[1],
                               point[0] + self.PIXEL_SIZE,
                               point[1] + self.PIXEL_SIZE)

    def update(self):
        self.text.config(text=str(self.count))
        self.set_rect_coords(self.input_paths[self.count], self.input_rects)

        if not self.pause and self.count < len(self.input_paths):
            self.count += 1
            self.master.after(self.CONFIG['step_delay'], self.update)
