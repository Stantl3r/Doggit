from build import *

train_data = load_train_data()
model = load_model()

train = train_data[:-3500]
test = train_data[-3500:]

X = np.array([i[0] for i in train]).reshape(-1,size,size,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,size,size,1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=30, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=500, show_metric=True, run_id=model_name)
model.save(model_name)