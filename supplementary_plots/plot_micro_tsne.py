import tensorflow as tf
import numpy as np
from models.micro_classifier import get_dataset, get_model
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

plt.rcParams.update({'font.size': 18})
plt.rcParams["figure.figsize"] = (13, 10)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"


activities = {
    0: 'laptop-typing',
    1: 'sitting',
    2: 'phone-typing',
    3: 'phone-talking',
    4: 'playing-guitar',
    5: 'eating-food',
    6: 'combing-hair',
    7: 'brushing',
    8: 'drinking-water'
}

model = get_model()
model.load_weights('micro_weights.h5')
print(model.summary())
X_train, X_test, y_train, y_test = get_dataset()
y_test = np.argmax(y_test,axis=1)
activity_labels = [activities[label] for label in y_test]

print('inp', y_test.shape)
embedding_model = tf.keras.Model(inputs=model.input,outputs=model.get_layer('global_average_pooling2d').output)
embeddings = embedding_model.predict(X_test)

np.savez('embeddings_and_data_micro.npz', embeddings=embeddings, test_data=X_test, test_labels=y_test)
print(f"Embeddings saved to embeddings.npy. Shape of embeddings: {embeddings.shape}")

scaler = StandardScaler()
features_scaled = scaler.fit_transform(embeddings)

loc = 'upper right'
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(features_scaled)
print('tsne_shape', tsne_results.shape)

plt.figure(figsize=(11, 7))
scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=y_test, cmap='tab10')

# Custom colorbar
cbar = plt.colorbar(scatter, ticks=range(9))
cbar.set_ticks(np.arange(9) + 0.5)
cbar.set_ticklabels([activities[i] for i in range(9)])

# plt.title('t-SNE of Test Data', fontsize=22, fontweight='bold')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.grid()
plt.tight_layout()
plt.savefig('micro_tsne.pdf')
plt.show()