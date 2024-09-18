import tensorflow as tf

def load_and_save_model():
    model = tf.keras.applications.MobileNetV2(weights='imagenet')
    # Save the model in HDF5 format
    model.save('mobilenetv2_local_model.h5')

if __name__ == "__main__":
    load_and_save_model()
