import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Paths
# -----------------------------
DATASET_DIR = "data"  # test 資料夾
test_dir = os.path.join(DATASET_DIR, "test")
MODEL_PATH = "cifar10_cnn_deploy.h5"  # 或 cifar10_cnn_deploy.h5

# -----------------------------
# Prepare test data
# -----------------------------
img_size = (128, 128)
batch_size = 32

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  
)

# -----------------------------
# Load model
# -----------------------------
model = load_model(MODEL_PATH)
print(model.summary())

# -----------------------------
# Predict
# -----------------------------
pred_probs = model.predict(test_generator)
y_pred = np.argmax(pred_probs, axis=1)
y_true = test_generator.classes
class_names = list(test_generator.class_indices.keys())

# -----------------------------
# Accuracy
# -----------------------------
accuracy = np.sum(y_pred == y_true) / len(y_true)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig("visualization/cnn_test_confusion_matrix.png")
plt.close()

# -----------------------------
# Classification Report
# -----------------------------
report = classification_report(y_true, y_pred, target_names=class_names)
print("\nClassification Report:\n")
print(report)
print("Confusion matrix saved to visualization/cnn_test_confusion_matrix.png")