import matplotlib.pyplot as plt
import numpy as np

def funcs(func_number: int):
    func1 = lambda x: np.where(x<2, 2, 0)

    func2 = lambda x: np.where(x < np.pi/2, 1.2*np.cos(x), 0)

    func3 = lambda x: 0.5 *np.arctan(x+2) + np.cos(x) + 0.1

    func4 = lambda x: np.where((0.3 < x) & (x < 0.3 + np.pi), np.sin(x - 0.3),  0)

    func = [func1, func2, func3, func4]

    return func[func_number]

x = np.linspace(0, 5, 100)

_, axes = plt.subplots(2, 2)

for i in range(2):
    for j in range(2):
        index = 2 * i + j
        y = funcs(index)(x)
        axes[i, j].plot(x, y)
        axes[i, j].fill_between(x, y, color= 'red', alpha = 0.5)
        axes[i, j].set_title(f'func{index}')
plt.show()
