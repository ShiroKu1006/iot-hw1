import os
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import plot_model

# -----------------------------
# Dataset paths
# -----------------------------
DATASET_DIR = "data"  # CIFAR-10 已轉成圖片的資料夾
train_dir = os.path.join(DATASET_DIR, "train")
test_dir = os.path.join(DATASET_DIR, "test")

# -----------------------------
# Image augmentation / preprocessing
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32
img_size = (128, 128)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

num_classes = train_generator.num_classes
print("Detected classes:", num_classes)

# -----------------------------
# Build CNN model
# -----------------------------
model = Sequential([
    # Conv block 1
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(128,128,3)),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    
    # Conv block 2
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    
    # Conv block 3
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    
    # Conv block 4
    Conv2D(256, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    # Conv block 5
    Conv2D(512, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    
    # 使用 Flatten 
    Flatten(), 
    # 全連接層
    Dense(256, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.summary()

model.compile(
    optimizer=Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# Callbacks
# -----------------------------
os.makedirs("visualization", exist_ok=True)

checkpoint = ModelCheckpoint(
    "cifar10_cnn_best.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# Save model architecture diagram
plot_model(model, to_file="visualization/cnn_architecture.png", show_shapes=True, show_layer_names=True)

# -----------------------------
# Train model
# -----------------------------
EPOCHS = 5
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=[checkpoint, early_stop],
    workers=32,
)

# -----------------------------
# Save final deploy model
# -----------------------------
model.save("cifar10_cnn_deploy.h5", include_optimizer=False) # for HUB 8735 ultra

best_model = tf.keras.models.load_model("cifar10_cnn_best.h5")

best_model.save(
    "cifar10_cnn_best.h5",
    include_optimizer=False
)

# -----------------------------
# Plot training curves
# -----------------------------
plt.figure()
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("visualization/cnn_training_accuracy.png")
plt.close()

plt.figure()
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("visualization/cnn_training_loss.png")
plt.close()

print("Training completed!")
print("- Best weights: cifar10_cnn_best.h5")
print("- Deploy model: cifar10_cnn_deploy.h5")
print("- Network graph: visualization/cnn_architecture.png")
print("- Training curves:")
print("    - visualization/cnn_training_accuracy.png")
print("    - visualization/cnn_training_loss.png") 