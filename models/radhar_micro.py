import pandas as pd
import numpy as np
from baseline.RadHAR.DataPreprocessing.voxels import voxalize
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from baseline.RadHAR.Classifiers.TD_CNN_LSTM import full_3D_model
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18})
plt.rcParams["figure.figsize"] = (13, 10)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

lbl_map = {'Clapping': 0,
           'jumping': 1,
           'lunges': 2,
           'running': 3,
           'squats': 4,
           'walking': 5,
           'waving': 6,
           'vaccum-cleaning': 7,
           'folding-clothes': 8,
           'changing-clothes': 9
}

lbl_map = {'laptop-typing': 0,
               'sitting': 1,
               'phone-typing': 2,
               'phone-talking': 3,
               'playing-guitar': 4,
               'eating-food': 5,
               'combing-hair': 6,
               'brushing': 7,
               'drinking-water': 8
               }


def get_points(): 
    df = pd.read_pickle('../datasets/micro_df_subset.pkl')
    points_data = df[['x_coord', 'y_coord', 'Activity']].values
    x_points = []
    y_points = []
    label = []
    data = []
    for points in points_data:
        xs = points[0]
        ys = points[1]
        label_val = points[2]
        data.append([xs,ys])
        label.append(label_val)

    # data = np.array(data)
    label = np.array(label)

    together_frames = 1
    sliding_frames = 1
    total_frames = len(data)
    i = 0
    j = 0
    data_pro1 = dict()
    while together_frames+i < total_frames:
        curr_j_data =[]
        for k in range(together_frames):
            curr_j_data = curr_j_data + data[i+k]
        #print(len(curr_j_data))
        data_pro1[j] = curr_j_data
        j = j+1
        i = i+sliding_frames

    pixels = []

    for i in data_pro1:
        f = data_pro1[i]
        f = np.array(f)
        x_c = f[0]
        y_c = f[1]
        
        pix = voxalize(10, 32, x_c, y_c)
        #print(i, f.shape,pix.shape)
        pixels.append(pix)

    pixels = np.array(pixels)
    frames_together = 60
    sliding = 10

    train_data=[]
    gt = []
    i = 0
    while i+frames_together<=pixels.shape[0]:
        local_data=[]
        for j in range(frames_together):
            local_data.append(pixels[i+j])
        gt.append(label[i+j])
        train_data.append(local_data)
        i = i + sliding
    max_index = label.shape[0] - frames_together
    train_data = np.array(train_data)
    gt = np.array(gt)
    y = to_categorical(np.array(list(map(lambda e: lbl_map[e], gt))), num_classes=9)

    X_train, X_test, y_train, y_test = train_test_split(train_data, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = get_points()
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

X_train = np.expand_dims(X_train,-1)
X_test = np.expand_dims(X_test,-1)
model = full_3D_model(X_train,y_train)

model.compile(loss="categorical_crossentropy", optimizer='adam',metrics="accuracy")


model.fit(
    X_train,
    y_train,
    epochs=100,
    validation_split=0.2,
    batch_size=32)

#model.load_weights('macro_weights.h5')

pred = model.predict([X_test])
conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(pred, axis=1))
total = conf_matrix / conf_matrix.sum(axis=1).reshape(-1, 1)
total = np.round(total, 2)
# labels = ['clapping', 'jumping', 'lunges', 'running', 'squats', 'walking', 'waving',
#         'vaccum\ncleaning', 'folding\nclothes', 'changing\nclothes']
labels = ['laptop\ntyping', 'sitting', 'phone\ntyping', 'phone\ntalking', 'playing\nguitar', 'eating\nfood',
            'combing\nhair', 'brushing', 'drinking\nwater']
df_cm = pd.DataFrame(total, index=[i for i in labels], columns=[i for i in labels])
sns.heatmap(df_cm, vmin=0, vmax=1, annot=True, cmap="Blues")
plt.xticks(rotation=35)
plt.yticks(rotation=35)
plt.tight_layout()
plt.savefig('radhar_classification_points.pdf')
plt.show()
print(classification_report(np.argmax(y_test, axis=1), np.argmax(pred, axis=1)))
    
