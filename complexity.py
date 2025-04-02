import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.stats import entropy
import pickle

# slicename = "E2MeV_L1um"
# slicename = "E1MeV_L001um"
slicename = "E7MeV_L001um"

fpkl = "toymc_example_"+slicename+".pkl"
with open(fpkl,'rb') as handle:
    data = pickle.load(handle)
    # print(data)

def compute_complexity(name, x, y):
    # 1. Compute curvature
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**(3/2)
    curvature = np.nan_to_num(curvature)  # Handle division by zero
    min_curvature = np.min(curvature)
    max_curvature = np.max(curvature)
    mean_curvature = np.mean(curvature)
    std_curvature = np.std(curvature)
    
    # 2. Fourier spectrum complexity
    y_fft = np.abs(fft(y))
    power_spectrum = y_fft**2
    high_freq_fraction = np.sum(power_spectrum[len(power_spectrum)//4:]) / np.sum(power_spectrum)
    
    # 3. Fractal dimension (box counting method)
    def box_counting(x, y, scales):
        counts = []
        for scale in scales:
            bins = np.arange(min(x), max(x) + scale, scale)
            bin_counts, _ = np.histogram(x, bins)
            counts.append(np.count_nonzero(bin_counts))
        return counts
    
    scales = np.logspace(np.log10(np.min(np.diff(x))), np.log10(np.max(x) - np.min(x)), num=10)
    counts = box_counting(x, y, scales)
    coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
    fractal_dimension = -coeffs[0]
    
    # 4. Total variation
    total_variation = np.sum(np.abs(np.diff(y)))
    
    # 5. Entropy of derivative distribution
    dy_dx = np.gradient(y, x)
    hist, _ = np.histogram(dy_dx, bins=50, density=True)
    hist = hist[hist > 0]  # Remove zero values to avoid log issues
    derivative_entropy = entropy(hist)
    
    return {
        # "min_curvature": min_curvature,
        # "max_curvature": max_curvature,
        "mean_curvature": mean_curvature,
        "std_curvature": std_curvature,
        "high_freq_fraction": high_freq_fraction,
        "fractal_dimension": fractal_dimension,
        "total_variation": total_variation,
        "derivative_entropy": derivative_entropy
    }

# Example usage:
# x = np.linspace(0, 10, 1000)
# y = np.sin(x) + np.random.normal(0, 0.1, len(x))
# print(compute_complexity("name",x, y))

x = data["x"]
y = data["y"]["hModel"]
complexity = compute_complexity(slicename,x, y)

# Plot the shape
fig = plt.figure(figsize=(8, 4))
ax  = fig.add_subplot()
ax.set_ylim(ymin=1e-10)
# plt.plot(x, y, label=slicename)
plt.plot(x, y)
plt.xlabel(r'$\Delta E_{\rm cnt}$ [MeV]')
plt.ylabel('Steps')
plt.title(slicename)
plt.legend()
plt.grid()
plt.yscale('log')
plt.xscale('log')
# plt.show()

xmid = x[0] + (x[-1]-x[0])/2.
x_range = (x[-1]-x[0])
ymid = np.min(y) + (np.max(y)-np.min(y))/2.
y_range = (np.max(y)-np.min(y))
count = 1
for cpx,val in complexity.items():
    ax.text(x[0]+0.000000000001*x_range, (ymid/500)/count, f'{cpx}: {val:.2E}', fontsize=10)
    count *= 10

plt.savefig(f"{slicename}.pdf")



