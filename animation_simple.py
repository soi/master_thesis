#! /usr/bin/env pytho)
from __future__ import print_function, division
try:
    import tkinter as tk
except ImportError:
    import Tkinter as tk # python2
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from terminaltables import AsciiTable
from subprocess import call

class Animation(tk.Frame):
    WINDOW_SIZE_Y = 750
    WINDOW_SIZE_X = int(WINDOW_SIZE_Y / 1.777777)
    PIXEL_SIZE = 6
    CONFIG = {
            'bg_color': '#ffffff',
            'step_delay': 400,
            'step_offset': 40,
            'plot_only_mse_overall': False,
            'plot_only_mse': True,
            'input': {
                'visible': True,
                'color': 'green'
            },
            'label': {
                'visible': True,
                'color': 'blue'
            },
            'pred': {
                'visible': True,
                'color': 'red'
            }}

    def __init__(self, input_paths, label_paths, rel_label_paths,
                 mean_rel, std_rel, rel_pred_paths):
        self.input_paths = input_paths
        self.label_paths = label_paths
        self.rel_label_paths = rel_label_paths
        self.mean_rel = mean_rel
        self.std_rel = std_rel
        self.rel_pred_paths = rel_pred_paths

        self.input_len = len(input_paths[0])
        self.output_len = len(label_paths[0])

        self.master = tk.Tk()
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.full_rect, self.canvas = self.init_canvas(self.CONFIG['bg_color'])
        self.input_rects = self.init_rects(self.input_len,
                                          self.CONFIG['input']['color'],
                                          fade_to_last=False)
        self.label_rects = self.init_rects(self.output_len,
                                           self.CONFIG['label']['color'],
                                           fade_to_last=True)
        self.pred_rects = self.init_rects(self.output_len,
                                          self.CONFIG['pred']['color'],
                                          fade_to_last=True)

        # GUI elements
        self.button_frame = tk.Frame(self.master)

        self.show_pred_button = tk.Button(self.button_frame, text='Show Pred')
        self.show_pred_button.bind('<ButtonPress-1>', self.show_pred)
        self.show_pred_button.pack(side='left')

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

        toggle = lambda: self.set_rect_visibility('input', self.input_rects)
        self.input_box = tk.Checkbutton(self.button_frame, text='input',
                                        command=toggle)
        self.input_box.pack(side='left')
        self.input_box.select()

        toggle = lambda: self.set_rect_visibility('label', self.label_rects)
        self.label_box = tk.Checkbutton(self.button_frame, text='label',
                                        command=toggle)
        self.label_box.pack(side='left')
        self.label_box.select()

        toggle = lambda: self.set_rect_visibility('pred', self.pred_rects)
        self.pred_box = tk.Checkbutton(self.button_frame, text='pred',
                                        command=toggle)
        self.pred_box.pack(side='left')
        self.pred_box.select()

        self.text = tk.Label(self.master, text=0)
        self.text.pack(side='bottom')

        self.input_text = tk.Label(self.master, text='input',
                fg=self.CONFIG['input']['color'])
        self.input_text.pack(side='right')

        self.label_text = tk.Label(self.master, text='label',
                fg=self.CONFIG['label']['color'])
        self.label_text.pack(side='right')

        self.pred_text = tk.Label(self.master, text='simple',
                fg=self.CONFIG['pred']['color'])
        self.pred_text.pack(side='right')

        self.delay_text = tk.Label(self.master, text=self.CONFIG['step_delay'],
                fg='black')
        self.delay_text.pack(side='right')

        # create plot axis
        if self.CONFIG['plot_only_mse']:
            self.fig, self.ax_mse = plt.subplots(figsize=(9, 4), nrows=1)
        elif self.CONFIG['plot_only_mse_overall']:
            self.fig, self.ax_mse_overall = plt.subplots(figsize=(9, 4), nrows=1)
            self.ax_mse_overall.set_ylabel('MSE', fontsize=13)
            self.ax_mse_overall.set_xlabel('Index in Predicted Sequence',
                                           fontsize=13)
            self.ax_mse_overall.tick_params(axis='both', which='major', labelsize=11)
            self.ax_mse_overall.set_title('Eights Dataset', fontsize=18)

            se = (self.rel_pred_paths - self.rel_label_paths) ** 2
            self.ax_mse_overall.plot(np.mean(se, axis=(0,2)), linewidth=2.0)
        else:
            self.fig, axes = plt.subplots(figsize=(11, 10), nrows=2, sharex=True)
            self.ax_mse = axes[0]
            self.ax_mse_overall = axes[1]

        # start from the train/test boundary
        self.count = 0
        self.pause = True
        self.update()
        self.master.mainloop()

    def on_closing(self):
        plt.close(self.fig)
        self.master.destroy()

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
                if i == 0:
                    self.canvas.itemconfig(rect, fill='#770000')
                else:
                    color_str = '#ff' + 2 * color_str_part
                    self.canvas.itemconfig(rect, fill=color_str)
            if color is 'green':
                if i == 0:
                    self.canvas.itemconfig(rect, fill='#007700')
                else:
                    color_str = '#' + color_str_part + 'ff' + color_str_part
                    self.canvas.itemconfig(rect, fill=color_str)
            if color is 'blue':
                if i == 0:
                    self.canvas.itemconfig(rect, fill='#000077')
                else:
                    color_str = '#' + color_str_part * 2 + 'ff'
                    self.canvas.itemconfig(rect, fill=color_str)
            color_int += offset

    def show_pred(self, event):
        labels = self.rel_label_paths[self.count]
        print('label')
        for i, l in enumerate(labels):
            if abs(np.max(l)) > 0.2:
                print('* ' + str(l), end='')
            else:
                print(l, end='')
            print(' ' + str(i))
        print('')

        pred = self.rel_pred_paths[self.count]
        print('prediction')
        for i, p in enumerate(pred):
            if abs(np.max(p)) > 0.2:
                print('* ' + str(p), end='')
            else:
                print(p, end='')
            print(' ' + str(i))
        print('')

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
        self.canvas.postscript(file="saved_canvas" + str(unix_time) + ".ps",
                               colormode='color')
        print("saved canvas image")
        plt.savefig("saved_plots" + str(unix_time) + ".eps", format="eps", dpi=1000)
        print("saved plot image")

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

    def update_plot(self):
        if self.CONFIG['plot_only_mse']:
            self.ax_mse.clear()
            self.ax_mse.set_ylabel('MSE', fontsize=12)
            self.ax_mse.set_xlabel('Index in Predicted Sequence', fontsize=12)
            self.ax_mse.set_title('Basic LSTM', fontsize=16)

            # plot mse over sequences length
            se = (self.rel_pred_paths[self.count] - self.rel_label_paths[self.count]) ** 2
            self.ax_mse.plot(np.mean(se, axis=1))
            print('mse this pred: ' + str(np.mean(se)))
        self.fig.tight_layout()
        self.fig.show()

    def rel_to_abs_pred(self):
        # undo standardization
        path = np.copy(self.rel_pred_paths[self.count])
        path *= self.std_rel
        path += self.mean_rel

        # convert to abs coordinates
        tmp = np.zeros((path.shape[0] + 1, 2))
        tmp[1:] = path
        tmp[0] = self.input_paths[self.count][-1]
        tmp = np.cumsum(tmp, axis=0)
        # remove starting point and wrap around
        path = tmp[1:] % 1.0
        return path

    def update(self):
        self.text.config(text=str(self.count))
        self.set_rect_coords(self.input_paths[self.count], self.input_rects)
        if self.CONFIG['label']['visible']:
            label = self.label_paths[self.count]
            self.set_rect_coords(label, self.label_rects)

        if self.CONFIG['pred']['visible']:
            self.set_rect_coords(self.rel_to_abs_pred(), self.pred_rects)

        self.update_plot()

        if not self.pause and self.count < len(self.input_paths):
            self.count += 1
            self.master.after(self.CONFIG['step_delay'], self.update)
