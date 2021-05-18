#!/usr/bin/env python
'''This scripts computes averages and mins. And returns the best configurations.'''

def get_stats(exp):
    with open(f'results_exp_{exp}/results.csv','r') as res:
        res = tuple(line.split(',') for line in res.read().split('\n') if line)

    td = 0 # total dnns
    tc = 0 # total cnns
    md = 1 # minimum dnns
    mc = 1 # minimum cnns
    bd = None # best dnn
    bc = None #best cnn

    for line in res:
        if line[1] == 'dense':
            v = float(line[4])
            if v < md:
                md = v
                bd = line
            td += v
        elif line[1] == 'conv':
            v = float(line[4])
            if v < mc:
                mc = v
                bc = line
            tc += v

    td /= 75
    tc /= 75

    td = round(td,2)
    md = round(md,2)
    tc = round(tc,2)
    mc = round(mc,2)

    return td,md,bd,tc,mc,bc

print('Experiment 1:')
print('''Average dnn loss: {}, minimum dnn loss: {}, best dnn: {}
Average cnn loss: {}, minimum cnn loss: {}, best cnn: {}'''.format(*get_stats(1)))
print('Experiment 2:')
print('''Average dnn loss: {}, minimum dnn loss: {}, best dnn: {}
Average cnn loss: {}, minimum cnn loss: {}, best cnn: {}'''.format(*get_stats(2)))
