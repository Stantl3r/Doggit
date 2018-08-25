from build import *

model = load_model()

train = train_data[:-400]
test = train_data[-400:]

X = np.array([i[0] for i in train]).reshape(-1,size,size,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,size,size,1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=20, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=500, show_metric=True, run_id=model_name)
model.save(model_name)