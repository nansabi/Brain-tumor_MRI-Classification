import os
import numpy as np
import tensorflow as tf

def densenet_preprocess(x):
    return tf.keras.applications.densenet.preprocess_input(x)


from PIL import Image

# Validation script to align model predictions locally with the original Colab state
MODEL_FILE = "densenet121_final.h5"
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

if os.path.exists(MODEL_FILE):
    print("Loading cached DenseNet121 model...")
    model = tf.keras.models.load_model(MODEL_FILE, compile=False, safe_mode=False, custom_objects={"densenet_preprocess": densenet_preprocess})
    
    # Generate dummy input to verify layers execution
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    pred = model.predict(dummy_input, verbose=0)[0]
    print("Dummy Input Check Successful. Softmax Output Length:", len(pred))
    print("Target Classes Map:", CLASS_NAMES)
else:
    print(f"[Error] Model file {MODEL_FILE} not found in the local path. Skipping alignment test.")
