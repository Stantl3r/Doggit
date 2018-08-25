import matplotlib.pyplot as plt
from build import *

model = load_model()

fig=plt.figure()

for num,data in enumerate(test_data[:18]):

	image_num = data[1]
	image_data = data[0]

	y = fig.add_subplot(4,5,num+1)
	orig = image_data
	data = image_data.reshape(size,size,1)
	model_out = model.predict([data])[0]

	if np.argmax(model_out) == 0: 
		str_label='German'
	elif np.argmax(model_out) == 1:
		str_label='Golden'
	elif np.argmax(model_out) == 2:
		str_label = 'Bulldog'
	elif np.argmax(model_out) == 3:
		str_label = 'Beagle'
	elif np.argmax(model_out) == 4:
		str_label = 'Corgi'
	"""elif np.argmax(model_out) == 5:
		str_label = 'Yorkshire'"""

	y.imshow(orig,cmap='gray')
	plt.title(str_label)
	y.axes.get_xaxis().set_visible(False)
	y.axes.get_yaxis().set_visible(False)
plt.show()