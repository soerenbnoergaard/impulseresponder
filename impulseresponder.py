import matplotlib
matplotlib.use("TkAgg")
import tkinter as tk

from numpy import *
from matplotlib.pyplot import *
from scipy import signal

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

        fr_input = tk.LabelFrame(self, text="Input")
        fr_output = tk.LabelFrame(self, text="Output")
        fr_plot = tk.LabelFrame(self, text="Plot")

        # INPUT
        tk.Label(fr_input, text="Sample rate [Hz]").grid(row=0, column=0)
        tk.OptionMenu(fr_input, self.var.sample_rate_Hz, "48000", command=lambda x: self.on_update_inputs()).grid(row=0, column=1)
        tk.Label(fr_input, text="Waveform").grid(row=1, column=0)
        tk.OptionMenu(fr_input, self.var.waveform, "prbs9", "prbs15", "prbs20", command=lambda x: self.on_update_inputs()).grid(row=1, column=1)
        tk.Button(fr_input, text="Save WAV", command=self.on_input_save_wav).grid(row=2, column=0, columnspan=2, sticky=tk.W+tk.E)

        # OUTPUT
        tk.Button(fr_output, text="Measure", command=self.on_measure).grid(row=0, column=0, columnspan=2, sticky=tk.W+tk.E)
        tk.Label(fr_output, text="Impulse response").grid(row = 1, column=0, columnspan=2, sticky=tk.W+tk.E)
        tk.Label(fr_output, text="Length [s]").grid(row=2, column=0)
        e = tk.Entry(fr_output, textvariable=self.var.impulse_response_length_s)
        e.grid(row=2, column=1)
        e.bind("<Return>", lambda x: self.on_update_impulse_response_length())
        tk.Button(fr_output, text="Save WAV", command=self.on_output_save_impulse_response_as_wav).grid(row=3, column=0, columnspan=2, sticky=tk.W+tk.E)

        # FRAMES
        fr_input.grid(row = 0, column = 0, sticky=tk.N)
        fr_output.grid(row = 0, column = 1, sticky=tk.N)
        fr_plot.grid(row = 1, column = 0, columnspan=2, sticky=tk.N)

        self.on_update_inputs()

    def on_measure(self):
        print("Not implemented yet")

    def on_input_save_wav(self):
        print("Not implemented yet")

    def on_update_inputs(self):
        fs = float(self.var.sample_rate_Hz.get())
        L = float(self.prbs_length(self.var.waveform.get()))

        self.var.impulse_response_length_s.set(str(L / fs))

    def on_update_impulse_response_length(self):
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

def _test_gui():
    root = tk.Tk()
    Gui(root).pack()
    root.mainloop()

f_sample = 48_000
f_nyquist = f_sample/2

# Estimate the impulse responsee settling time
settling_time_s = 0.01

def main():
    # Generate test filter
    b, a = get_filter_coefficients()

    # Make measurement on pseudo-random noise
    # x = prbs9(2**9-1)
    x = prbs15(2**15-1)
    # x = prbs20(2**20-1)
    # x = prbs23(2**23-1)
    y = signal.lfilter(b, a, x)

    # Estimate impulse response
    t, h = get_impulse_response(b, a)
    h_ = signal.correlate(y, x, "full")
    h_ = h_[len(h_)//2:]
    h_ *= sum(h)/sum(h_[0:len(t)])
    h_ = h_[0:int(f_sample * settling_time_s)]
    t_ = linspace(0, len(h_)/f_sample, len(h_))

    # Estimate frequency response
    f, H = get_freq_response(b, a)
    H_ = fft.fft(h_)
    f_ = linspace(0, f_sample, len(H_))

    # Show results
    fig, (ax1, ax2) = subplots(2, 1, figsize=[10, 7])

    plot_impulse_response(t, h, ax=ax1, label="$h$")
    plot_impulse_response(t_, h_, ax=ax1, label="$h_{est}$")

    plot_freq_response(f, H, ax=ax2, label="$H$")
    plot_freq_response(f_, H_, ax=ax2, label="$H_{est}$")

    decorate_plots(fig, ax1, ax2)

def get_filter_coefficients():
    return signal.cheby1(4, 10, 10_000/f_nyquist)

def get_impulse_response(b, a):
    t, h = signal.dimpulse([b, a, 1/f_sample])
    h = h[0]
    return t, h

def get_freq_response(b, a):
    w, H = signal.freqz(b, a)
    f = w*f_sample/(2*pi)
    return f, H

def plot_impulse_response(t, h, ax=None, label=None):
    if ax is None:
        ax = gca()
    ax.plot(t, h, label=label)

def plot_freq_response(f, H, ax=None, label=None):
    if ax is None:
        ax = gca()
    ax.plot(f, 20*log10(abs(H)), label=label)

def decorate_plots(fig, ax1, ax2):
    for ax in [ax1, ax2]:
        ax.legend()
        ax.grid(True, "both", "both")
    
    ax1.set_xlabel("Time [s]")
    ax2.set_xlabel("Frequency [Hz]")

    ax1.set_ylabel("Impulse reponse")
    ax2.set_ylabel("Frequency response [dB]")

    ax2.set_xlim(0, f_sample/2)
    ax2.set_ylim(-60, 5)

    fig.tight_layout()
    fig.savefig("audio_impulse_response_tool.png")

def prbs_generic(num_bits, seed, field_width, newbit_function):
    mask = int("1" * field_width, 2)
    a = seed
    i = 1
    bits = []
    for _ in range(num_bits):
        newbit = newbit_function(a)
        a = ((a << 1) | newbit) & mask;
        bits.append(newbit * 2 - 1) # NRZ sequence
    return bits

def prbs7(num_bits, seed=0x01):
    return prbs_generic(
        num_bits,
        seed,
        7,
        lambda x: (((x >> 6) ^ (x >> 5)) & 1)
    )

def prbs9(num_bits, seed=0x01):
    return prbs_generic(
        num_bits,
        seed,
        9,
        lambda x: (((x >> 8) ^ (x >> 4)) & 1)
    )

def prbs15(num_bits, seed=0x01):
    return prbs_generic(
        num_bits,
        seed,
        15,
        lambda x: (((x >> 14) ^ (x >> 13)) & 1)
    )

def prbs20(num_bits, seed=0x01):
    return prbs_generic(
        num_bits,
        seed,
        20,
        lambda x: (((x >> 19) ^ (x >> 2)) & 1)
    )

def prbs23(num_bits, seed=0x01):
    return prbs_generic(
        num_bits,
        seed,
        23,
        lambda x: (((x >> 22) ^ (x >> 17)) & 1)
    )

def _test_prbs():
    for f, n in [
                [prbs7, 7],
                [prbs9, 9],
                [prbs15, 15],
                [prbs20, 20],
                [prbs23, 23],
            ]:
        N = 2**n - 1
        x = array(f(N)) * 2 - 1

        R = signal.correlate(x, x, "full")
        R = R/max(R)
        X = linspace(-1, 1, len(R))
        plot(X, R)


if __name__ == "__main__":
    _test_gui()
    # _test_prbs()
    # main()
    # legend()
    show()
