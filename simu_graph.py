import matplotlib.pyplot as plt
import numpy as np

new_f = lambda x: np.where(x < 2.5, 2/3*((-np.arctan(10*(x-1.7)) + 1.2)/4+ 0.0616104), 2/3*((np.arctan(10 *(x-3.3)) + 1.2)/4+ 0.0616104))

def funcs(func_number: int):
    func1 = lambda x: np.where(x<1, 1, 0)

    func2 = lambda x: np.where(x < np.pi/2, 1.2*np.cos(x), 0)

    func3 = lambda x: np.where(x < np.pi/1.8, np.cos(0.9* x), 0)

    func4 = lambda x: np.where(x < np.pi/0.6, 0.6 * np.cos(0.6*x),  0)

    func5 = lambda x: (-0.5 * np.arctan(4*x-6)+0.785) * 0.7

    func6 = lambda x: -0.5 * np.arctan(4*x-6)+0.785

    func7 = lambda x: 1.5/np.sqrt(2* np.pi* 0.17) * np.exp(- np.square(x- 2.5)/(2 * 0.17))

    func8 = lambda x: 1.5/np.sqrt(2* np.pi* 0.5) * np.exp(- np.square(x- 2)/(2 * 0.5))

    func = [func1, func2, func3, func4, func5, func6, func7, func8]

    return func[func_number]


x = np.linspace(0, 5, 100)
additional_plot = new_f(x)


# plot for funcs
_, axes = plt.subplots(2, 4, figsize= (10, 5))

for i in range(2):
    for j in range(4):
        index = 4 * i + j
        y = funcs(index)(x)
        axes[i, j].plot(x, y)
        if index in [6]:
            axes[i, j].plot(x, additional_plot, linestyle = '--')
        axes[i, j].arrow(0, 0, 0, 2.5)
        axes[i, j].arrow(0, 0, 5.5, 0)
        if index not in [4, 5]:
            axes[i, j].fill_between(x, y, color= 'blue', alpha = 0.5)
        axes[i, j].set_ylim([0, 2])
        axes[i, j].set_xlim([0, 5])
        axes[i, j].set_yticks([i/2 for i in range(5)])
        axes[i, j].set_xticks([i for i in range(6)])
        # axes[i, j].set_title(f'func{index}')
        axes[i, j].set_frame_on(False)
plt.savefig('results/sim.png')
plt.show()
