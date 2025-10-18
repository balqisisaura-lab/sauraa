import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Pastikan versi library aman
print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

# 2. Set default dtype yang valid
tf.keras.backend.set_floatx('float32')

# 3. Jika kamu punya model lama atau pretrained, hapus atau re-train saja
# Misal modelnya bernama model, pastikan semua layer pakai dtype yang valid
def build_model(input_shape):
    inputs = keras.Input(shape=input_shape, dtype='float32')  # jangan tuple!
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 4. Contoh data dummy untuk uji coba
X_train = np.random.rand(100, 10).astype('float32')
y_train = np.random.randint(0, 2, size=(100, 1)).astype('float32')

# 5. Bangun dan latih model
model = build_model((10,))
model.fit(X_train, y_train, epochs=3, batch_size=8)

# 6. Simpan model (opsional)
model.save("model_safe.keras")

# 7. Jika kamu jalankan ini di dashboard.py, pastikan tidak ada variabel Keras
#    yang punya dtype tuple, misalnya:
#    ❌ salah: keras.Variable((1, 2), dtype=(tf.float32, tf.int32))
#    ✅ benar: keras.Variable([1.0, 2.0], dtype='float32')

print("✅ Model berhasil dijalankan tanpa error dtype tuple!")
