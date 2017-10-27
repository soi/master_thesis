#! /usr/bin/env python
from __future__ import print_function, division
try:
    import tkinter as tk
except ImportError:
    import Tkinter as tk # python2
import numpy as np
import random

class Animation(tk.Frame):
    WINDOW_SIZE = 800
    PIXEL_SIZE = 6
    CONFIG = {
            'bg_color': '#ffffff',
            'step_delay': 400,
            'step_offset': 40,
            'orig': {
                'visible': True,
                'color': 'blue'
            },
            'decoded': {
                'visible': True,
                'color': 'red'
            }}

    def __init__(self, paths, train_len, latent, decoded, plotting_ax):
        self.paths = paths
        self.train_len = train_len
        self.latent= latent
        self.decoded = decoded
        self.input_len = len(paths[0])
        self.ax = plotting_ax

        self.master = tk.Tk()
        self.full_rect, self.canvas = self.init_canvas(self.CONFIG['bg_color'])
        self.orig_rects = self.init_rects(self.input_len,
                                          self.CONFIG['orig']['color'])
        self.decoded_rects = self.init_rects(self.input_len,
                             self.CONFIG['decoded']['color'])

        # GUI elements
        self.button_frame = tk.Frame(self.master)
        self.go_to_start_button = tk.Button(self.button_frame, text='Go to start')
        self.go_to_start_button.pack(side='left')
        self.go_to_start_button.bind('<ButtonPress-1>', self.go_to_start)

        self.test_start_button = tk.Button(self.button_frame, text='Test start')
        self.test_start_button.pack(side='left')
        self.test_start_button.bind('<ButtonPress-1>', self.go_to_test_start)

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

        toggle = lambda: self.set_rect_visibility('orig', self.orig_rects)
        self.orig_box = tk.Checkbutton(self.button_frame, text='orig',
                                        command=toggle)
        self.orig_box.pack(side='left')
        self.orig_box.select()

        toggle = lambda: self.set_rect_visibility('decoded', self.decoded_rects)
        self.decoded_box = tk.Checkbutton(self.button_frame, text='decoded',
                                        command=toggle)
        self.decoded_box.pack(side='left')
        self.decoded_box.select()

        self.text = tk.Label(self.master, text=0)
        self.text.pack(side='bottom')

        self.latent_text = tk.Label(self.master, text=0)
        self.latent_text.pack(side='left')

        self.orig_text = tk.Label(self.master, text='orig',
                fg=self.CONFIG['orig']['color'])
        self.orig_text.pack(side='right')

        self.decoded_text = tk.Label(self.master, text='decoded',
                fg=self.CONFIG['decoded']['color'])
        self.decoded_text.pack(side='right')

        self.delay_text = tk.Label(self.master, text=self.CONFIG['step_delay'],
                fg='black')
        self.delay_text.pack(side='right')

        # start from the train/test boundary
        self.count = self.train_len
        self.pause = False
        self.update()
        self.master.mainloop()

    def init_canvas(self, bg_color):
        canvas = tk.Canvas(self.master, height=self.WINDOW_SIZE,
                width=self.WINDOW_SIZE)
        full_rect = canvas.create_rectangle(0, 0, self.WINDOW_SIZE,
                self.WINDOW_SIZE, outline='', fill=bg_color)
        canvas.pack()
        return full_rect, canvas

    def init_rects(self, count, color):
        rects = []
        for _ in range(count):
            rect = self.canvas.create_rectangle(0, 0, 1, 1, outline='')
            rects.append(rect)
        self.change_rect_color(rects, color)
        return rects

    def set_rect_visibility(self, rect_name, rects, toggle=True, value=True):
        """When toggle is True, value is ignored"""
        if toggle:
            self.CONFIG[rect_name]['visible'] = not self.CONFIG[rect_name]['visible']
        else:
            self.CONFIG[rect_name]['visible'] = False
        for rect in rects:
            if self.CONFIG[rect_name]['visible']:
                self.canvas.itemconfig(rect, state='normal')
            else:
                self.canvas.itemconfig(rect, state='hidden')

    def change_rect_color(self, rects, color, clear=False):
        if clear:
            for rect in rects:
                self.canvas.itemconfig(rect, fill=self.CONFIG['bg_color'])
        else:
            offset = 0
            color_int = 0
            for rect in rects:
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
                color_int -= offset

    def clear(self):
        self.canvas.tag_raise(self.full_rect)

    def go_to_start(self, event):
        print('start')
        self.pause = True
        self.count = 0
        self.update()

    def go_to_test_start(self, event):
        print('test start')
        self.pause = True
        self.count = self.train_len
        self.update()

    def go_back(self, event):
        print('back')
        self.pause = True
        self.count -= 1
        self.update()

    def go_forward(self, event):
        print('forward')
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
        self.CONFIG['step_delay'] = max(20, new_delay)
        self.delay_text.config(text=self.CONFIG['step_delay'])
        print('faster', self.CONFIG['step_delay'])

    def slower(self, event):
        self.CONFIG['step_delay'] += self.CONFIG['step_offset']
        self.delay_text.config(text=self.CONFIG['step_delay'])
        print('slower', self.CONFIG['step_delay'])

    def save(self, event):
        unix_time = time.time()
        self.canvas.postscript(file="saved_canvas" + str(unix_time) + ".ps",
                               colormode='color')
        print("saved canvas image")
        # plt.savefig("saved_plots" + str(unix_time) + ".eps", format="eps", dpi=1000)
        # print("saved plot image")

    def update_scatter(self):
        test_count = self.count - self.train_len
        lt = self.latent.T
        self.ax.clear()
        self.ax.scatter(lt[0], lt[1])
        # self.ax.scatter(lt[0][test_count], lt[1][test_count])

    def get_coord(self, data_value):
        max_size = self.WINDOW_SIZE / self.PIXEL_SIZE
        return int(data_value * max_size) * self.PIXEL_SIZE

    def check_data_availability(self):
        if self.count >= self.train_len:
            self.decoded_box.config(state='normal')
            self.text.config(text=str(self.count), fg='blue')
        else:
            self.set_rect_visibility('decoded', self.decoded_rects, toggle=False,
                                     value=False)
            self.decoded_box.config(state='disabled')
            self.text.config(text=str(self.count), fg='black')

    def set_rect_coords(self, data, rects):
        for i, data_point in enumerate(data):
            self.canvas.coords(rects[i],
                    self.get_coord(data_point[0]),
                    self.get_coord(data_point[1]),
                    self.get_coord(data_point[0]) + self.PIXEL_SIZE,
                    self.get_coord(data_point[1]) + self.PIXEL_SIZE)

    def update(self):
        self.check_data_availability()
        self.set_rect_coords(self.paths[self.count], self.orig_rects)

        if self.CONFIG['decoded']['visible']:
            decoded = self.decoded[self.count - self.train_len]
            self.set_rect_coords(decoded, self.decoded_rects)

            latent_space = self.latent[self.count-self.train_len]
            latent_str = '  '.join(['%.2f' % e for e in latent_space])
            self.latent_text.config(text=latent_str)

        if self.ax is not None:
            self.update_scatter()

        if not self.pause and self.count < len(self.paths):
            self.count += 1
            self.master.after(self.CONFIG['step_delay'], self.update)

if __name__ == '__main__':
    pass
