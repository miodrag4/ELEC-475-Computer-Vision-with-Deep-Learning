import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.interpolate import make_interp_spline

# Loss values from the training output
loss_values = [
    3.9396966028091547,
3.3049474860091346,
2.9784592838238573,
2.7035970986651643,
2.446219739401737,
2.2478827863093227,
2.0905780453816094,
1.9526998975392802,
1.8419174846175992,
1.7301099336970494
]

# Epoch numbers
epochs = range(1, 11)

xnew = np.linspace(min(epochs), max(epochs), 300)
spl = make_interp_spline(epochs, loss_values, k=3)
power_smooth = spl(xnew)

# Plot the training loss curve
plt.figure(figsize=(10, 5))
plt.plot(xnew, power_smooth, label='Training Loss', color='blue', linewidth=2)

# Adding titles and labels
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Display the legend
plt.legend()

plt.show()