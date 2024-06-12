import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
import datetime
tf.random.set_seed(32)
np.random.seed(32)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import seaborn as sns
plt.rcParams.update({'font.size': 18})
plt.rcParams["figure.figsize"] = (13, 10)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"


def scale(doppz, Max=9343, Min=36240):
    doppz_scaled = (doppz - Min) / (Max - Min)
    return doppz_scaled


def StackFrames(doppz, labels, frame_stack=10):
    max_index = doppz.shape[0] - frame_stack
    stacked_doppz = np.array([doppz[i:i + frame_stack] for i in range(max_index)]).transpose(0, 2, 3, 1)
    new_labels = np.array([labels[i + frame_stack - 1] for i in range(max_index)])
    return stacked_doppz, new_labels


class Dataset:
    def __init__(self, loc="../datasets/processed_datasets/macro_df_subset.pkl", frame_stack=1):
        print(f"loading dataset from {loc}")
        df = pd.read_pickle(loc)
        df = df[df.Activity != '  '].reset_index()
        doppz = np.array(df['doppz'].values.tolist())
        label = df['Activity'].values
        dop_max, dop_min = doppz.max(), doppz.min()
        doppz_scaled_stacked, new_labels = StackFrames(scale(doppz, dop_max, dop_min), label, frame_stack)
        self.data, self.label = self.process(doppz_scaled_stacked, new_labels)

    def process(self, doppz_scaled_stacked, new_labels):
        return doppz_scaled_stacked, new_labels


def get_dataset():
    data = Dataset(loc='../datasets/processed_datasets/macro_df_subset.pkl',
                   frame_stack=1)

    lbl_map = {'Clapping': 0,
               'jumping': 1,
               'lunges': 2,
               'running': 3,
               'squats': 4,
               'walking': 5,
               'waving': 6,
               'vaccum-cleaning': 7,
               'folding-clothes': 8,
               'changing-clothes': 9}

    X_norm = data.data
    y = to_categorical(np.array(list(map(lambda e: lbl_map[e], data.label))), num_classes=10)

    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def get_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (2, 5), (1, 2), padding="same", activation='relu', input_shape=(16, 256, 1)),
        tf.keras.layers.Conv2D(64, (2, 3), (1, 2), padding="same", activation='relu'),
        tf.keras.layers.Conv2D(96, (2, 3), (1, 2), padding="same", activation='relu'),
        tf.keras.layers.Conv2D(96, (2, 3), (1, 2), padding="same", activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, "relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, "softmax")
    ])
    return model

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_dataset()

    model = get_model()
    print(model.summary())
    model.compile(loss="categorical_crossentropy", optimizer='adam',metrics="accuracy")

    folder=datetime.datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
    best_save=tf.keras.callbacks.ModelCheckpoint(filepath='macro_weights.h5',save_weights_only=True,
                                                    monitor='val_accuracy',mode='max',save_best_only=True)
    tbd=tf.keras.callbacks.TensorBoard(log_dir=f'logs/{folder}')

    model.fit(
        X_train,
        y_train,
        epochs=100,
        validation_split=0.2,
        batch_size=32,
        callbacks=[best_save,tbd])

    #model.load_weights('macro_weights.h5')

    pred = model.predict([X_test])
    conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(pred, axis=1))
    total = conf_matrix / conf_matrix.sum(axis=1).reshape(-1, 1)
    total = np.round(total, 2)
    labels = ['clapping', 'jumping', 'lunges', 'running', 'squats', 'walking', 'waving',
            'vaccum\ncleaning', 'folding\nclothes', 'changing\nclothes']
    df_cm = pd.DataFrame(total, index=[i for i in labels], columns=[i for i in labels])
    sns.heatmap(df_cm, vmin=0, vmax=1, annot=True, cmap="Blues")
    plt.xticks(rotation=35)
    plt.yticks(rotation=35)
    plt.tight_layout()
    plt.savefig('macro_classification.pdf')
    plt.show()
    print(classification_report(np.argmax(y_test, axis=1), np.argmax(pred, axis=1)))
        
