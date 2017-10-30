#! /usr/bin/env python
from __future__ import print_function, division
try:
    import tkinter as tk
except ImportError:
    import Tkinter as tk # python2
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import multivariate_normal
from terminaltables import AsciiTable

class Animation(tk.Frame):
    WINDOW_SIZE_Y = 750
    WINDOW_SIZE_X = int(WINDOW_SIZE_Y / 1.777777)
    PIXEL_SIZE = 6
    CONFIG = {
            'bg_color': '#ffffff',
            'step_delay': 400,
            'step_offset': 40,
            'plot_contour': True,
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
                 mean_rel, std_rel, rel_pred_paths, mdn_params, component_choice):
        self.input_paths = input_paths
        self.label_paths = label_paths
        self.rel_label_paths = rel_label_paths
        self.mean_rel = mean_rel
        self.std_rel = std_rel
        self.rel_pred_paths = rel_pred_paths
        self.mdn_params = mdn_params
        self.component_choice = component_choice

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

        self.pause_button = tk.Button(self.button_frame, text='Play')
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

        self.input_text = tk.Label(self.master, text='Input',
                fg=self.CONFIG['input']['color'])
        self.input_text.pack(side='right')

        self.label_text = tk.Label(self.master, text='Label',
                fg=self.CONFIG['label']['color'])
        self.label_text.pack(side='right')

        self.pred_text = tk.Label(self.master, text='Prediction',
                fg=self.CONFIG['pred']['color'])
        self.pred_text.pack(side='right')

        self.delay_text = tk.Label(self.master, text=self.CONFIG['step_delay'],
                fg='black')
        self.delay_text.pack(side='right')

        # create contour plot axes
        if self.CONFIG['plot_contour']:
            self.contour_fig, self.ax_co = plt.subplots(1, figsize=(5, 5 * 1.7777))
            self.ax_co.invert_yaxis()
            self.ax_co.get_xaxis().set_visible(False)
            self.ax_co.get_yaxis().set_visible(False)

            # colorbar
            divider = make_axes_locatable(self.ax_co)
            self.ax_co_cb = divider.append_axes("right", size="5%", pad=0.1)

        # other plots
        self.graphs_fig, axes = plt.subplots(figsize=(11, 10), nrows=3, sharex=True)
        self.ax_pi = axes[0]
        self.ax_mse = axes[1]
        self.ax_mse_overall = axes[2]
        self.ax_mse_overall.set_title('Overall MSE', fontsize=12)
        self.ax_mse_overall.set_ylabel('MSE', fontsize=11)
        self.ax_mse_overall.set_xlabel('Index in Predicted Sequence', fontsize=11)

        # plot mse overall as it never changes
        se = (self.rel_pred_paths - self.rel_label_paths) ** 2
        self.ax_mse_overall.plot(np.mean(se, axis=(0,2)))

        # start from the train/test boundary
        self.count = 0
        self.pause = True
        self.update()
        self.master.mainloop()

    def on_closing(self):
        if self.CONFIG['plot_contour']:
            plt.close(self.contour_fig)
        plt.close(self.graphs_fig)
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

    def preprocess_current_mdn(self):
        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

        # splitting and preprocessing
        mdn_copy = np.copy(self.mdn_params[self.count])
        pi, mu1, mu2, sig1, sig2, corr = np.split(mdn_copy, 6, axis=1)
        for i in range(len(pi)):
            pi[i] = softmax(pi[i])
        sig1 = np.exp(sig1)
        sig2 = np.exp(sig2)
        corr = np.tanh(corr)
        return pi, mu1, mu2, sig1, sig2, corr

    def print_mdn(self):
        def _(x):
            return "{0:.4f}".format(x)

        pi, mu1, mu2, sig1, sig2, corr = self.preprocess_current_mdn()

        # one table per point with one distribution per line
        # creating the tables
        for i in range(len(self.rel_pred_paths[0])):
            if abs(np.max(self.rel_pred_paths[self.count, i])) > 0.2:
                name = 'Point ' + str(i) + ' (jump)'
            else:
                name = 'Point ' + str(i)
            table = [[name, 'pi', 'mu1', 'sig1', 'mu2', 'sig2', 'corr']]
            component = self.component_choice[self.count, i]

            print()
            print(self.rel_pred_paths[self.count, i])

            # creating one table for a single point prediction
            for j in range(len(pi[0])):
                if pi[i, j] > 0.3:
                    number = '*' + str(j) + '*'
                else:
                    number = str(j)
                if j == component:
                    number += '!'
                table.append([number,
                              _(pi[i, j]),
                              _(mu1[i, j]),
                              _(sig1[i, j]),
                              _(mu2[i, j]),
                              _(sig2[i, j]),
                              _(corr[i, j])])
            t = AsciiTable(table)
            t.justify_columns = {0: 'center'}
            print(t.table)

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

        start_point = self.input_paths[self.count][-1]
        print('start_point: ' + str(start_point))

        self.print_mdn()

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
        self.graphs_fig.savefig("saved_plots" + str(unix_time) + ".eps",
                                format="eps",
                                dpi=1000)
        print("saved plot image")
        if self.CONFIG['plot_contour']:
            self.contour_fig.savefig("saved_contour" + str(unix_time) + ".png",
                                      bbox_inches='tight',
                                      format="png",
                                      dpi=700)
            print("saved contour image")

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

    def tick_formatter(self, val, pos=None):
        return val + 1

    def update_contour_plot(self):
        self.ax_co.clear()
        self.ax_co.set_title('Values of the cumulative GMMs', fontsize=12)

        pi, mu1, mu2, sig1, sig2, corr = self.preprocess_current_mdn()
        resolution = 100 # equal for each dim

        # construct cumsum of label to shift prediction space
        unscaled_rel_pred_paths = self.rel_pred_paths[self.count] * self.std_rel
        start_point = np.zeros((len(pi) + 1, 2))
        start_point[0] = self.input_paths[self.count][-1]
        start_point[1:] = unscaled_rel_pred_paths
        start_point = np.cumsum(start_point, axis=0)[1:] % 1.0
        result = np.zeros((resolution, resolution))

        # iterate over predicted point length
        for p in range(len(pi)):
            # starting space
            x = np.arange(0.0, 1.0, 1 / resolution)
            y = np.arange(0.0, 1.0, 1 / resolution)
            # move starting point to 0,0
            x -= start_point[p][0]
            y -= start_point[p][1]
            # scale space to the prediction space
            x *= 1 / self.std_rel[0]
            y *= 1 / self.std_rel[1]
            X, Y = np.meshgrid(x, y)
            pos = np.empty(X.shape + (2,))
            pos[:, :, 0] = X; pos[:, :, 1] = Y
            # sum up all bivare Gaussians
            for i in range(len(pi[p])):
                m1 = mu1[p][i]
                m2 = mu2[p][i]
                s1 = sig1[p][i]
                s2 = sig2[p][i]
                c = corr[p][i]
                mean = [m1, m2]
                cov = [[s1*s1, c*s1*s2], [c*s1*s2, s2*s2]]
                rv = multivariate_normal(mean, cov)
                result += pi[p][i] * rv.pdf(pos)

        # undo scaling and translation
        X *= self.std_rel[0]
        Y *= self.std_rel[1]
        X += start_point[p][0]
        Y += start_point[p][1]

        result += 1e-10
        result = np.log10(result)
        cs = self.ax_co.contourf(X, Y, result, 30)

        # colorbar
        self.ax_co_cb.clear()
        self.colorbar = self.contour_fig.colorbar(cs, cax=self.ax_co_cb)
        self.colorbar.set_clim(vmin=np.min(result), vmax=np.max(result))
        self.colorbar.draw_all()

        # show everything
        self.contour_fig.show()

    def update_graphs_plots(self):
        self.ax_pi.clear()
        self.ax_pi.set_title(r'$\Pi$ values of the GMM for this prediction', fontsize=12)
        self.ax_pi.set_ylabel(r'$\pi$', fontsize=14)

        pi, mu1, mu2, sig1, sig2, corr = self.preprocess_current_mdn()
        # preparing component choices
        choices = self.component_choice[self.count].astype(np.int16)
        # one hot encode component choice for masking
        choice_mask = np.ones(pi.shape)
        choice_mask[np.arange(len(choices)), choices] = 0
        pi_masked = np.ma.masked_array(pi, mask=choice_mask)

        # plot pi values
        for i, dist in enumerate(pi.T):
            self.ax_pi.plot(dist, color='C' + str(i))
            self.ax_pi.plot(pi_masked.T[i], color='C' + str(i), marker='o')

        # plot mse over sequences length
        self.ax_mse.clear()
        self.ax_mse.set_title('MSE of this prediction', fontsize=12)
        self.ax_mse.set_ylabel('MSE', fontsize=11)
        se = (self.rel_pred_paths[self.count] - self.rel_label_paths[self.count]) ** 2
        self.ax_mse.plot(np.mean(se, axis=1))
        print('mse this pred: ' + str(np.mean(se)))

        self.graphs_fig.tight_layout()
        self.graphs_fig.show()

    def update_plots(self):
        self.update_graphs_plots()
        if self.CONFIG['plot_contour']:
            self.update_contour_plot()

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
            # import pudb; pudb.set_trace()
            self.set_rect_coords(self.rel_to_abs_pred(), self.pred_rects)

        self.update_plots()

        if not self.pause and self.count < len(self.input_paths):
            self.count += 1
            self.master.after(self.CONFIG['step_delay'], self.update)
