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
 


import json
def get_training_data(path):
    training_sample = []
    with open(path) as trainfile:
        train_data = json.load(trainfile)
    training_support_data = []
    training_refute_data = []
    training_notenough_data = []
    for record in train_data:
        if train_data[record]['label']=='SUPPORTS':
            training_support_data.append(train_data[record])
        elif train_data[record]['label']=='REFUTES':
            training_refute_data.append(train_data[record])
        elif train_data[record]['label']=='NOT ENOUGH INFO':
            training_notenough_data.append(train_data[record])
    assert(len(train_data)==len(training_support_data)+len(training_refute_data)+len(training_notenough_data))
    print("--------------------Distribution in training dataset-----------------------------------------")
    print('Percent of support entries %.3f %%'%(100*(len(training_support_data)/len(train_data))))
    print('Percent of refute entries %.3f %%'%(100*(len(training_refute_data)/len(train_data))))
    print('Percent of not-enough entries %.3f %%'%(100*(len(training_notenough_data)/len(train_data))))
    print("---------------------------------------------------------------------------------------------")
    #1.
    temp=[]
    temp2={}
    refute = {}
    support = {}
    MAX_LENGTH_OF_EVIDENCES = 25
    for record in training_refute_data:
        temp.append(len(record['evidence']))
    for i in range(1,MAX_LENGTH_OF_EVIDENCES+1):#Note:the max number of evidences for SUPPORT/REFUTE is 48
        temp2[i]=0
    temp = sorted(temp)
    temp = temp[0:temp.index(MAX_LENGTH_OF_EVIDENCES+1)]
    for i in temp:
        temp2[i]+=1
    refute =temp2
    #refute is a dictionary where key=length of evidences and value=no of evidences whose length is as indicated by key
    #------------------------------------------------------------------------------------------------------------
    temp=[]
    temp2={}
    for record in training_support_data:
        temp.append(len(record['evidence']))
    for i in range(1,MAX_LENGTH_OF_EVIDENCES+1):#Note:the max number of evidences for SUPPORT/REFUTE is 48
        temp2[i]=0
    temp = sorted(temp)
    temp = temp[0:temp.index(MAX_LENGTH_OF_EVIDENCES+1)]
    for i in temp:
        temp2[i]+=1
    support=temp2
    #support is a dictionary where key=length of evidences and value=no of evidences whose length is as indicated by key
    #2
    import numpy as np
    def get_indices_and_shuffle(dataset):
        length_indicies = {}#dictionary...key=length of evidences value:list of indices
        for i in list(range(1,MAX_LENGTH_OF_EVIDENCES+1)):
            temp=list()
            length_indicies[i]=temp
        for record in dataset:
            if len(record['evidence']) in list(range(1,26)):
                temp_list = length_indicies[len(record['evidence'])]
                temp_list.append(dataset.index(record))
        return length_indicies
    length_indicies_support = get_indices_and_shuffle(training_support_data)
    length_indicies_refute = get_indices_and_shuffle(training_refute_data)
    sample_not_enough = random.sample(training_notenough_data,2500)
    refute_indices  = shuffle_indices(length_indicies_refute,200)
    support_indices = shuffle_indices(length_indicies_support,100)
    for index in refute_indices:
        training_sample.append(training_refute_data[index])
    for index in support_indices:
        training_sample.append(training_support_data[index])
    training_sample.extend(sample_not_enough)
    np.random.shuffle(training_sample)
    assert(len(training_sample)==len(refute_indices)+len(support_indices)+len(sample_not_enough))
    return training_sample


import random
def shuffle_indices(length_indicies_dict,k):
    final_indices = []
    for length,index_list in length_indicies_dict.items():
        if(len(index_list)<=k):
            final_indices.extend(index_list)
        else:
            final_indices.extend(random.sample(index_list, k))
    if(len(final_indices)<2500):
        final_indices.extend(random.sample(length_indicies_dict[3],2500-len(final_indices)))
        final_indices = set(final_indices)
    np.random.shuffle(list(final_indices))
    return final_indices


