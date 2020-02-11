# fit mlp model on problem 2 and save model to file
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from matplotlib import pyplot
import category_encoders as ce


# Reference: https://machinelearningmastery.com/

# define and fit model on a training dataset
def fit_model(trainX, trainy, devX, devy):
    # define model
    model = Sequential()
    model.add(Dense(5, input_dim = 4, activation='relu',use_bias=True, kernel_initializer='he_uniform'))#Hidden layer with 5 units
    #'he_uniform' draws inital weights from uniform distribution
    model.add(Dense(3, activation='softmax'))
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit model
    history = model.fit(trainX, trainy, validation_data=(devX, devy), epochs=100, verbose=0,batch_size=128)
    return model, history

# summarize the performance of the fit model
def summarize_model(model, history, trainX, trainy, devX, devy):
    # evaluate the model
    _, train_acc = model.evaluate(trainX, trainy, verbose=0)
    _, dev_acc = model.evaluate(devX, devy, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, dev_acc))
    # plot loss during training
    pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='dev')
    pyplot.legend()
    # plot accuracy during training
    pyplot.subplot(212)
    pyplot.title('Accuracy')
    pyplot.plot(history.history['acc'], label='train')
    pyplot.plot(history.history['val_acc'], label='dev')
    pyplot.legend()
    pyplot.show()
from keras.models import load_model
# define and fit model on a training dataset
def refit_model(trainX, trainy, devX, devy):
    # load the model we had stored earlier
    # model trainded by 2000 claims in the traing data
    model = load_model('model_snli_large2.h5')

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit model
    history = model.fit(trainX, trainy, validation_data=(devX, devy), epochs=100, verbose=0,batch_size=128)
    return model, history

# summarize the performance of the fit model
def summarize_model(model, history, trainX, trainy, devX, devy):
    # evaluate the model
    _, train_acc = model.evaluate(trainX, trainy, verbose=0)
    _, dev_acc = model.evaluate(devX, devy, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, dev_acc))
    # plot loss during training
    pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='dev')
    pyplot.legend()
    # plot accuracy during training
    pyplot.subplot(212)
    pyplot.title('Accuracy')
    pyplot.plot(history.history['acc'], label='train')
    pyplot.plot(history.history['val_acc'], label='dev')
    pyplot.legend()
    pyplot.show()
  
  
