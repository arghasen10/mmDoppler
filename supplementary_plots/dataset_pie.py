import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 28})
plt.rcParams["figure.figsize"] = (13, 10)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

macro_df = pd.read_pickle('../datasets/macro_df.pkl')
micro_df = pd.read_pickle('../datasets/micro_df.pkl')

macro_acts = macro_df.groupby('activity').size()
macro_acts = (macro_acts/np.sum(macro_acts))*100 
micro_acts = micro_df.groupby('activity').size()
micro_acts = (micro_acts/np.sum(micro_acts))*100 

macro_act_names = ['Clapping', 'jumping', 'lunges', 'running', 'squats', 'walking', 'waving', 'vaccum\ncleaning', 'folding\nclothes', 
                   'changing\nclothes']

plt.pie(macro_acts, labels=macro_act_names, autopct='%1.1f%%')
plt.tight_layout()
plt.savefig('macro_distribution.pdf')
plt.cla()
micro_act_names = ['laptop-typing', 'sitting', 'phone\ntyping', 'phone\ntalking', 'playing\nguitar', 'eating\nfood', 'combing\nhair', 'brushing', 'drinking\nwater']
plt.pie(micro_acts, labels=micro_act_names, autopct='%1.1f%%')
plt.tight_layout()
plt.savefig('micro_distribution.pdf')
