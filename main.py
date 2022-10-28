import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.stats as st
import seaborn as sns
import csv


def approximation(x, y, d):
    fp, residuals, rank, sv, rcond = sp.polyfit(x, y, d, full=True)
    f = sp.poly1d(fp)  # аппроксимирующая функция
    print('Коэффициент -- a %s  ' % round(fp[0], 4))
    print('Коэффициент-- b %s  ' % round(fp[1], 4))
    print('Коэффициент -- c %s  ' % round(fp[2], 4))
    y1 = [fp[0] * x[i] ** 2 + fp[1] * x[i] + fp[2] for i in range(0, len(x))]
    so = round(sum([abs(y[i] - y1[i]) for i in range(0, len(x))]) / (len(x) * sum(y)) * 100, 4)
    print('Average quadratic deviation ' + str(so))
    fx = sp.linspace(x[0], x[-1] + 1, len(x))  # можно установить вместо len(x) большее число для интерполяции
    plt.plot(fx, f(fx), linewidth=2)

    plt.grid(True)


Filename = 'Data.xlsx'

plt.style.use('dark_background')
fig = plt.figure()

excel_data = pd.read_excel(Filename)
data = pd.DataFrame(excel_data, columns=['x', 'y'])

x = []
y = []
i = 0
d = 3 #степень полинома

for column in data['x']:
    x.append(float(column))
    y.append(float(data['y'][i]))
    i = i + 1


fp = np.polyfit(x, y, d)
f = np.poly1d(fp)
t = np.linspace(min(x), max(x), len(x))
t2 = np.linspace(min(y), max(y), len(y))
ci = 0.95 * np.std(y) / np.mean(y)
t1 = st.t.interval(alpha=0.95, df=len(y)-1, loc=np.mean(y), scale=st.sem(y))

print('Интервал: ', ci)

plt.xlabel("название оси")
plt.ylabel("название оси")
plt.title('Diabetes Dataset')

sns.regplot(x=x,
            y=y,
            order=d,
            ci=ci,
            scatter_kws={'s': 1},
            line_kws={"lw": 1})

plt.plot(x, y, color='yellow', linestyle=' ', marker='+')
plt.legend(loc=0)


approximation(x, y, d)
plt.savefig('figure')

f = str(f)
data_file = open('text.txt', 'w')
data_file.write('function =' + '\n')
data_file.write(f + '\n')
data_file.write('t:' + '\n')
for lines in t:
    data_file.write(str(lines)+ '\n')
data_file.write('ci' + '\n')
data_file.write(str(ci) + '\n')
data_file.write('t1:' + '\n')
data_file.write(str(t1) + '\n')


