from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow import keras
import numpy as np

arrayOfName = ["airplane" ,"automobile" ,"bird" ,"cat" ,"deer" ,"dog", "frog", "horse", "ship", "truck"]
# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, target_size=(32, 32))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape(1, 32, 32, 3)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img


# It can be used to reconstruct the model identically.
reconstructed_model = keras.models.load_model("my_model")

# Load CIFAR-10 data
(input_train, target_train), (input_test, target_test) = cifar10.load_data()

# Generate generalization metrics
score = reconstructed_model.evaluate(input_test, target_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
# predict the class
img = load_image('sample_image.png')
result = np.argmax(reconstructed_model.predict(img), axis=-1)
print(arrayOfName[result[0]])
img = load_image('deer.png')
result = np.argmax(reconstructed_model.predict(img), axis=-1)
print(arrayOfName[result[0]])
img = load_image('unknown.jpg')
result = np.argmax(reconstructed_model.predict(img), axis=-1)
print(arrayOfName[result[0]])
img = load_image('truck.jpeg')
result = np.argmax(reconstructed_model.predict(img), axis=-1)
print(arrayOfName[result[0]])
img = load_image('bird6.png')
result = np.argmax(reconstructed_model.predict(img), axis=-1)
print(arrayOfName[result[0]])
