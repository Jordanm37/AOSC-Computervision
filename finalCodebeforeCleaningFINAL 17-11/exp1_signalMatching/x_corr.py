import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

def mfreqz(b, a=1):
    w, h = signal.freqz(b,a)
    h_dB = 20 * np.log10 (abs(h))
    plt.subplot(211)
    plt.plot(w / max(w), h_dB)
    plt.ylim(-150, 5)
    plt.ylabel('Magnitude (db)')
    plt.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    plt.title(r'Frequency response')
    plt.subplot(212)
    h_Phase = np.unwrap(np.arctan2(np.imag(h), np.real(h)))
    plt.plot(w/max(w),h_Phase)
    plt.ylabel('Phase (radians)')
    plt.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    plt.title(r'Phase response')
    plt.subplots_adjust(hspace=0.5)
 
def plot_graphs(x1,x2,x3):
    fig = plt.figure()
    plt.plot(x1, 'b', label='$x_1$')
    plt.plot(x2, 'g', label='$x_2$: delayed')
    plt.plot(x3, 'r', label='$x_3$: advanced')
    plt.grid()
    plt.legend()
    plt.savefig('signals.jpg')
 
    fig = plt.figure(figsize=(10, 4))
    axes = fig.add_subplot(1, 2, 1)
    axes.plot(lags, crr12)
    axes.set_xlabel("lag (samples)")
    axes.set_ylabel("xcorr (x1, x2)")
    axes.grid()
    axes.set_title("delay: {:d} samples".format(delay12))
    axes = fig.add_subplot(1, 2, 2)
    axes.plot(lags, crr13)
    axes.set_xlabel("lag (samples)")
    axes.set_ylabel("xcorr (x1, x3)")
    axes.grid()
    axes.set_title("delay: {:d} samples".format(delay13))
    plt.savefig('xcorrs.jpg')
    plt.show()


nx = 50
D = 2

x1 = np.random.randn(nx)
x2 = np.roll(x1, D)
x3 = np.roll(x1, -D)

crr12 = signal.correlate(x1, x2)
crr13 = signal.correlate(x1, x3)

lags = np.arange(-nx + 1, nx)
delay12 = lags[np.argmax(crr12)]
delay13 = lags[np.argmax(crr13)]
plot_graphs(x1,x2,x3)
  
n = 201
fs = 32
ft = signal.firwin(n, cutoff = [1, 4], window = "hanning", pass_zero=False, fs=fs)
mfreqz(ft)
plt.show()