import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import csv
from sklearn.model_selection import train_test_split
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import seaborn as sns
from collections import Counter
import pickle
from imblearn.over_sampling import SMOTE

    
def find_annot(activity):
    annots = {
        'clapping':0, 
        'clappng':0,
        'juming':1, 
        'jumping':1, 
        'lunges':2,
        'lnges':2,
        'walking':3,
        'alking':3,
        'squats':4,
        'squat':4, 
        'waving':5,
        'foldingClothes':6,
        'foldCloth':6,
        'changingClothes':7, 
        'chnageCloth':7,
        'vacuumCleaning':8,
        'vacuum':8, 
        'running':9, 
        'Running':9,
        'phone_typing':10,
        'phonetyping':10,
        'laptop_typing':11, 
        'laptoptype':11,
        'sitting':12, 
        'eating':13, 
        'eating_food':13, 
        'eatting':13,
        'phone_talking':14,
        'phoone_talking':14,
        'phonetalk':14,
        'playing_guitar':15, 
        'guiter':15,
        'brushingTeeth': 16, 
        'brushing':16,
        'combing':17,
        'combingHair':17,
        'drinkingWater':18, 
        'drinking':18,
    }
    for act in annots.keys():
        if act in activity:
            return annots[act]
def map_macro_micro(a):
    a_map = \
    {0: 'macro',
     1: 'macro',
     2: 'macro',
     3: 'macro',
     4: 'macro',
     5: 'macro',
     6: 'macro',
     7: 'macro',
     8: 'macro',
     9: 'macro',
     10: 'micro',
     11: 'micro',
     12: 'micro',
     13: 'micro',
     14: 'micro',
     15: 'micro',
     16: 'micro',
     17: 'micro',
     18: 'micro'
    }
    return a_map[a]


def process_mmwave(f):
    u = f.split('/')[1].split('_')[0]
    data = [json.loads(val) for val in open(f, "r")]
    annot = find_annot(data[0]['activity'])
    if data[0]['datenow'].split('/')[1] == '0':
        new_date = '/'.join([data[0]['datenow'].split('/')[0],'1',data[0]['datenow'].split('/')[-1]])
        datetime_str = datetime.strftime(datetime.strptime(new_date, "%d/%m/%Y"), "%Y-%m-%d")+' '
    else:
        datetime_str = datetime.strftime(datetime.strptime(data[0]['datenow'], "%d/%m/%Y") + relativedelta(months=1), "%Y-%m-%d")+' '
    mmwave_df = pd.DataFrame.from_dict(data)
    mmwave_df['datetime'] = mmwave_df['timenow'].apply(lambda e: datetime_str + ':'.join(e.split('_')))
    mmwave_df['User'] = u
    mmwave_df['activity']=annot
    if 'doppz' in mmwave_df.columns:
        mmwave_df['doppz'] = list(np.array(mmwave_df['doppz'].values.tolist()))
        mmwave_df = mmwave_df[['datetime', 'rangeIdx', 'dopplerIdx', 'numDetectedObj', 'range', 'peakVal', 
                               'x_coord', 'y_coord', 'doppz', 'activity']]
    else:
        return
    mmwave_df['activity_class'] = mmwave_df['activity'].map(lambda x: map_macro_micro(x))
    return mmwave_df


def read_mmwave():
    mmwave_files = glob.glob('singleuser/*.txt')
    return pd.concat([process_mmwave(f) for f in mmwave_files], ignore_index=True)


mmwave_df = read_mmwave()


mmwave_df[mmwave_df.activity_class == 'micro'].to_pickle('micro_df.pkl')
mmwave_df[mmwave_df.activity_class == 'macro'].to_pickle('macro_df.pkl')