import matplotlib.pyplot as plt 
import numpy as np 
plt.rcParams.update({'font.size': 34})
plt.rcParams["figure.figsize"] = (13, 10)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"  
# create data 
x = np.arange(2) 
y1 = [0.95, 0.98] 
y2 = [0.81, 0.95] 
width = 0.2
  
# plot data in grouped manner of bar type 
plt.bar(x-0.1, y1, width, label='Macro') 
plt.bar(x+0.1, y2, width, label='Micro') 
plt.ylim(0,1.1)
plt.xticks(x, ['RadHAR', 'mmDoppler']) 
plt.ylabel("Accuracy") 
plt.legend(loc='upper right', ncol=2) 
plt.grid()
plt.tight_layout()
plt.savefig('comparison.pdf')
plt.show() 