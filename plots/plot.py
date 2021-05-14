import pandas as pd
import matplotlib.pyplot as plt

e1 = pd.read_csv('../results_exp_1/results.csv')
for i in range(0,24,4):
	data = e1.iloc[i:i+4]
	data.plot(\
		x='|E|',\
		y=['dijkstra','bellman_ford','moore'],\
		ylabel='time(ms)',\
		title=f'Time(ms) vs |E|: density={SP.iloc[i]["density"]}, delta={int(SP.iloc[i]["delta"])}',\
		alpha=0.8,
		linewidth=5)
print(e1)

e2 = pd.read_csv('../results_exp_2/results.csv')
for i in range(0,24,4):
	data = e2.iloc[i:i+4]
	data.plot(\
		x='|E|',\
		y=['bfs','bfs_bi','dijkstra','dijkstra_bi'],\
		ylabel='time(ms)',\
		title=f'Time(ms) vs |E|: density={SP.iloc[i]["density"]}, delta={int(SP.iloc[i]["delta"])}',\
		alpha=0.8,
		linewidth=5)
print(e2)

for i in plt.get_fignums():
    plt.figure(i)
    plt.savefig(f'fig{i}.pdf')
