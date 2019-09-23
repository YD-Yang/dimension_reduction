from sklearn.model_selection import train_test_split
x_train, x_test = train_test_split(X, test_size = 0.2, random_state = 1) 


x_input_shape = x_train.shape[1]

from keras.models import Sequential, Model 
from keras.layers import Dense 
from keras.optimizers import Adam 

model = Sequential()
model.add(Dense(512, activation = 'relu', input_shape = (x_input_shape,)))
model.add(Dense(256, activation = 'relu'))
model.add(Dense(256, activation = 'linear', name = 'bottleneck'))
model.add(Dense(256, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(x_input_shape, activation = 'sigmoid'))
model.compile(loss = 'mean_squared_error', optimizer = Adam())
history = model.fit(x_train, x_train, batch_size = 64, epochs = 3, validation_data = (x_test, x_test))
encoder = Model(model.input, model.get_layer('bottleneck').output)

X = pd.DataFrame(X)
X.columns = ['tfidf_encoder' + '_' + str(i) for i in X]
X = pd.concat([X, med[keycols].reset_index()], axis = 1)
X = X.drop('index', axis = 1)
