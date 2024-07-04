import os
from sklearn.decomposition import PCA
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

def load_images_from_folder(folder, label, image_size=(64, 64)):
    images = []
    labels = []
    file_list = os.listdir(folder)
    for filename in tqdm(file_list, desc=f"Loading {label} images"):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, image_size)
            img = img.flatten()
            images.append(img)
            labels.append(label)
    return images, labels


cat_images, cat_labels = load_images_from_folder('C:/Users/PC/Desktop/Prodigy Infotech Machine Learning Internship/Cat_Dog_Classification/dataset/train/cats', label=0)
dog_images, dog_labels = load_images_from_folder('C:/Users/PC/Desktop/Prodigy Infotech Machine Learning Internship/Cat_Dog_Classification/dataset/train/dogs', label=1)


images = np.array(cat_images + dog_images)
labels = np.array(cat_labels + dog_labels)

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

pca = PCA(n_components=100) 
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)

svm = SVC(kernel='linear')
svm.fit(X_train_pca, y_train)

y_val_pred = svm.predict(X_val_pca)

print("----------------------------------------------------------------------------------------------------------------------------------------------")
print("Metrics for Validation set :")
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Classification Report:")
print(classification_report(y_val, y_val_pred))
print("----------------------------------------------------------------------------------------------------------------------------------------------")


test_cat_images, test_cat_labels = load_images_from_folder('C:/Users/PC/Desktop/Prodigy Infotech Machine Learning Internship/Cat_Dog_Classification/dataset/test/cats', label=0)
test_dog_images, test_dog_labels = load_images_from_folder('C:/Users/PC/Desktop/Prodigy Infotech Machine Learning Internship/Cat_Dog_Classification/dataset/test/dogs', label=1)


test_images = np.array(test_cat_images + test_dog_images)
test_labels = np.array(test_cat_labels + test_dog_labels)

X_test_pca = pca.fit_transform(test_images)
y_test_pred = svm.predict(X_test_pca)

print("----------------------------------------------------------------------------------------------------------------------------------------------")
print("Metrics for Validation set :")
print("Validation Accuracy:", accuracy_score(test_labels, y_test_pred))
print("Classification Report:")
print(classification_report(test_labels, y_test_pred))
print("----------------------------------------------------------------------------------------------------------------------------------------------")