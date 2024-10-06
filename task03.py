import os
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

train_dir = "/content/train"
train_images = os.listdir(train_dir)[:5000]

features = []
labels = []
image_size = (50, 50)

for image in tqdm(train_images, desc="Processing Train Images"):
    label = 0 if image.startswith('cat') else 1
    image_read = cv2.imread(os.path.join(train_dir, image))
    image_resized = cv2.resize(image_read, image_size)
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    image_flattened = image_gray.flatten()
    image_normalized = image_flattened / 255.0
    features.append(image_normalized)
    labels.append(label)

features = np.array(features)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=True, random_state=42)

pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train_pca, y_train)

y_pred = svm_model.predict(X_test_pca)

target_names = ['Cat', 'Dog']
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
