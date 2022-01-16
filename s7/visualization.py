import matplotlib.pyplot as plt

num_workers = list(range(1, 5))
avg = [7.9522562503814695, 5.275395917892456, 4.868078804016113, 4.31802544593811]
std = [0.29203548680079583, 0.3589278593057674, 0.3373675071337621, 0.029915650502388827]

plt.errorbar(num_workers, avg, yerr=std)
plt.show()