import joblib
import os
import tensorflow as tf

def save_model(model, filepath):
    """Saves a model using joblib or Keras save format."""
    try:
        if isinstance(model, tf.keras.Model):
            tf.keras.models.save_model(model, filepath)
            print(f"Keras model saved successfully to {filepath}")
        else:
            joblib.dump(model, filepath)
            print(f"Object saved successfully using joblib to {filepath}")
    except Exception as e:
        print(f"Error saving model/object to {filepath}: {e}")

def load_model(filepath):
    """Loads a model using joblib or Keras load format."""
    try:
        if filepath.endswith(".keras") or filepath.endswith(".h5"):
            model = tf.keras.models.load_model(filepath)
            print(f"Keras model loaded successfully from {filepath}")
        else:
            model = joblib.load(filepath)
            print(f"Object loaded successfully using joblib from {filepath}")
        return model
    except FileNotFoundError:
        print(f"Error: Model/Object file not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading model/object from {filepath}: {e}")
        if "loading a Keras model" in str(e):
            print("Attempting joblib load instead...")
            try:
                model = joblib.load(filepath)
                print(f"Object loaded successfully using joblib from {filepath}")
                return model
            except Exception as e2:
                print(f"Joblib load also failed: {e2}")
                return None
        return None

def save_label_encoder(encoder, filepath):
    """Saves a LabelEncoder using joblib."""
    save_model(encoder, filepath)

def load_label_encoder(filepath):
    """Loads a LabelEncoder using joblib."""
    return load_model(filepath)