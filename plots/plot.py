#!/usr/bin/env python
'''This scripts generates plots for experiments 1 and 2.'''





import pandas as pd
import matplotlib.pyplot as plt





def loss_epochs(df,exp):
	'''Zero-One loss vs #Epochs'''
	_,ax = plt.subplots()
	colors = {'dense':'red','conv':'blue'}
	grouped = df.groupby('model_type')
	for key, group in grouped:
	    group.plot(ax=ax, kind='scatter', x='n_epochs', y='zero_one_loss', label=key, color=colors[key])
	plt.title(f'Exp{exp}: Zero-One loss vs #Epochs')
	plt.xlabel('# Epochs')
	plt.ylabel('Zero-One loss')
	plt.savefig(f'e{exp}_loss_epochs.pdf')
	plt.close()





def loss_width(df,exp):
	'''Zero-One loss vs Model Size'''
	colors = {1:'red',2:'blue',3:'green'}
	for i in range(10):
		image_size = df.iloc[i*15]['image_size']
		model_type = df.iloc[i*15]['model_type']
		_,ax = plt.subplots()
		grouped = df.iloc[i*15:(i+1)*15].groupby('model_depth')
		for key, group in grouped:
		    group.plot(ax=ax,kind='line',x='model_size',y='zero_one_loss',label=f'depth:{key}',color=colors[key])
		plt.title(f'Exp{exp}: Zero-One loss vs Model Width (image=({image_size}x{image_size}),type={model_type})')
		plt.xlabel('Model Width')
		plt.ylabel('Zero-One loss')
		plt.savefig(f'e{exp}_{image_size}_{model_type}_loss_width.pdf')
		plt.close()





def loss_image_size(df,exp):
	'''Zero-One loss vs Model Size'''
	colors = ['red','blue','green','orange','purple']
	for i in range(6):
		_,ax = plt.subplots()
		for j in range(5):
			p = i*5+j*30
			image_size = df.iloc[p]['image_size']
			model_depth = df.iloc[p]['model_depth']
			model_type = df.iloc[p]['model_type']
			data = df.iloc[p:p+5]
			data.plot(ax=ax,kind='line',x='model_size',y='zero_one_loss',label=f'i. size:{image_size}',color=colors[j])
		plt.title(f'Exp{exp}: Zero-One loss vs Model Width (depth={model_depth},type={model_type})')
		plt.xlabel('Model Width')
		plt.ylabel('Zero-One loss')
		plt.savefig(f'e{exp}_{model_depth}_{model_type}_loss_image_size.pdf')
		plt.close()





def plot_experiment(path,exp):
	'''Generate plots for experiment.'''
	df = pd.read_csv(path)
	print(df)
	loss_epochs(df,exp)
	loss_width(df,exp)
	loss_image_size(df,exp)





def main():
	plot_experiment('../results_exp_1/results.csv',1)
	plot_experiment('../results_exp_2/results.csv',2)





if __name__ == '__main__':
	main()
