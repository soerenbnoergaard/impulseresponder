import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt

from scipy import signal
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def main():
    # imp = ImpulseResponderSimulation(48000)
    # imp.measure("prbs15")
    # imp.analyze(0.01)

    main_gui()

def main_gui():
    root = tk.Tk()
    root.title("Impulse Responder")
    Gui(root).pack(expand=True, fill=tk.BOTH)
    root.mainloop()

class ImpulseResponder(object):
    def __init__(self, sample_rate_Hz):
        self.sample_rate_Hz = sample_rate_Hz

        self.x = None # Input
        self.y = None # Output
        self.h = None # Impulse response
        self.H = None # Frequency response
        self.t = None # Time axis
        self.f = None # Frequency axis

    def get_waveform_data(self, waveform_string):
        noise_function = {
            "prbs9": self.prbs9,
            "prbs15": self.prbs15,
            "prbs20": self.prbs20,
        }[waveform_string]

        length = self.get_prbs_length(waveform_string)
        return noise_function(length)

    def get_prbs_length(self, s):
        return {
            "prbs9": 2**9-1,
            "prbs15": 2**15-1,
            "prbs20": 2**20-1,
        }[s]

    def prbs_generic(self, num_bits, seed, field_width, newbit_function):
        mask = int("1" * field_width, 2)
        a = seed
        bits = []
        for _ in range(num_bits):
            newbit = newbit_function(a)
            a = ((a << 1) | newbit) & mask
            bits.append(newbit * 2 - 1) # NRZ sequence
        return bits

    def prbs7(self, num_bits, seed=0x01):
        return self.prbs_generic(num_bits, seed, 7, lambda x: (((x >> 6) ^ (x >> 5)) & 1))

    def prbs9(self, num_bits, seed=0x01):
        return self.prbs_generic(num_bits, seed, 9, lambda x: (((x >> 8) ^ (x >> 4)) & 1))

    def prbs15(self, num_bits, seed=0x01):
        return self.prbs_generic(num_bits, seed, 15, lambda x: (((x >> 14) ^ (x >> 13)) & 1))

    def prbs20(self, num_bits, seed=0x01):
        return self.prbs_generic(num_bits, seed, 20, lambda x: (((x >> 19) ^ (x >> 2)) & 1))

    def prbs23(self, num_bits, seed=0x01):
        return self.prbs_generic(num_bits, seed, 23, lambda x: (((x >> 22) ^ (x >> 17)) & 1))

    def analyze(self, impulse_response_length_s):
        # Estimate impulse response
        h = signal.correlate(self.y, self.x, "full")
        h /= len(self.x) # TODO: Is this scaling correct?
        h = h[len(h)//2:]
        h = h[0:int(impulse_response_length_s * self.sample_rate_Hz)]

        t = np.linspace(0, impulse_response_length_s, len(h))

        # Estimate frequency response
        H = np.fft.fft(h)
        H = H[0:len(H)//2] 
        f = np.linspace(0, self.sample_rate_Hz/2, len(H))

        # Store results
        self.h = h
        self.t = t
        self.H = H
        self.f = f

class ImpulseResponderSimulation(ImpulseResponder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.b, self.a = signal.cheby1(4, 10, 10_000 / (self.sample_rate_Hz/2))

    def measure(self, waveform_string):
        x = self.get_waveform_data(waveform_string)
        y = signal.lfilter(self.b, self.a, x)

        # Store results
        self.x = x
        self.y = y

class Var(object):
    def __init__(self):
        self.sample_rate_Hz = tk.StringVar()
        self.waveform = tk.StringVar()
        self.impulse_response_length_s = tk.StringVar()

class Gui(tk.Frame):
    def __init__(self, parent):
        self.parent = parent
        self.var = Var()

        self.var.sample_rate_Hz.set("48000")
        self.var.waveform.set("prbs15")
        self.var.impulse_response_length_s.set("1")

        tk.Frame.__init__(self, parent)

        # FRAMES
        fr_A = tk.Frame(self)
        fr_B = tk.Frame(self)

        fr_A.pack(side=tk.TOP, expand=False, fill=tk.X)
        fr_B.pack(side=tk.TOP, expand=True, fill=tk.BOTH)

        fr_input = tk.LabelFrame(fr_A, text="Input")
        fr_output = tk.LabelFrame(fr_A, text="Output")
        fr_plot = tk.Frame(fr_B)

        fr_input.pack(side=tk.LEFT)
        fr_output.pack(side=tk.LEFT)
        fr_plot.pack(side=tk.TOP, expand=True, fill=tk.BOTH)

        # INPUT
        tk.Label(fr_input, text="Sample rate [Hz]").grid(row=0, column=0)
        tk.OptionMenu(fr_input, self.var.sample_rate_Hz, "48000", command=lambda x: self.on_update_input()).grid(row=0, column=1)
        tk.Label(fr_input, text="Waveform").grid(row=1, column=0)
        tk.OptionMenu(fr_input, self.var.waveform, "prbs9", "prbs15", "prbs20", command=lambda x: self.on_update_input()).grid(row=1, column=1)
        tk.Button(fr_input, text="Save WAV", command=self.on_input_save_wav).grid(row=2, column=0, columnspan=2, sticky=tk.W+tk.E)

        # OUTPUT
        tk.Button(fr_output, text="Measure", command=self.on_measure).grid(row=0, column=0, columnspan=2, sticky=tk.W+tk.E)
        tk.Label(fr_output, text="Impulse response").grid(row = 1, column=0, columnspan=2, sticky=tk.W+tk.E)
        tk.Label(fr_output, text="Length [s]").grid(row=2, column=0)
        e = tk.Entry(fr_output, textvariable=self.var.impulse_response_length_s)
        e.grid(row=2, column=1)
        e.bind("<Return>", lambda x: self.on_update_output())
        tk.Button(fr_output, text="Save WAV", command=self.on_output_save_impulse_response_as_wav).grid(row=3, column=0, columnspan=2, sticky=tk.W+tk.E)

        # PLOT
        self.fig = plt.Figure(dpi=100)
        canvas = FigureCanvasTkAgg(self.fig, master=fr_plot)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

        self.on_update_input()

    def on_measure(self):
        self.meas = ImpulseResponderSimulation(float(self.var.sample_rate_Hz.get()))
        self.meas.measure(self.var.waveform.get())
        self.on_update_output()

    def on_update_input(self):
        fs = float(self.var.sample_rate_Hz.get())
        L = float(self.prbs_length(self.var.waveform.get()))

        self.var.impulse_response_length_s.set(str(L / fs))

    def on_update_output(self):
        self.meas.analyze(float(self.var.impulse_response_length_s.get()))

        # Show results
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[10, 7])

        self.fig.clear()
        ax1 = self.fig.add_subplot(2, 1, 1)
        ax2 = self.fig.add_subplot(2, 1, 2)

        ax1.plot(self.meas.t, self.meas.h)
        ax2.plot(self.meas.f, 20*np.log10(abs(self.meas.H)))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def on_input_save_wav(self):
        print("Not implemented yet")

    def on_output_save_recording_as_wav(self):
        print("Not implemented yet")

    def on_output_save_impulse_response_as_wav(self):
        print("Not implemented yet")

    def prbs_length(self, s):
        return {
            "prbs9": 2**9-1,
            "prbs15": 2**15-1,
            "prbs20": 2**20-1,
        }[s]

if __name__ == "__main__":
    main()
