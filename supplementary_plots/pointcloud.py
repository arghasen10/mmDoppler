import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import chain

plt.rcParams.update({'font.size': 32})
plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

df = pd.read_pickle('../datasets/processed_datasets/macro_df_subset.pkl')

x_coord = df[df.Activity == 'walking']['x_coord'].values.tolist()
x_coord = np.array(list(chain(*x_coord)))

y_coord = df[df.Activity == 'walking']['y_coord'].values.tolist()
y_coord = np.array(list(chain(*y_coord)))


plt.scatter(x_coord[35:1036], y_coord[35:1036])
plt.xlim(-5,5)
plt.ylim(0.1,5)

plt.tight_layout()
plt.grid()
plt.savefig('pointcloud.pdf')