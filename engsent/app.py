from flask import Flask, request, jsonify, json
from flask_restful import Resource, Api
from flask_cors import CORS

import re
import warnings

from sklearn import metrics
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
import os
import pickle
import time
import numpy as np

from asyncio.windows_events import NULL
from underthesea import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, GlobalAveragePooling1D, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, SpatialDropout1D
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences


app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
api = Api(app)

stop_word_arr = []
dict = []
with open('datacmt.txt', "r", encoding="UTF-8") as f:
    sentences = f.readlines()

with open("vietnamese-stopwords.txt", "r", encoding="UTF-8") as fi:
    stop_word = fi.readlines()
    for j in stop_word:
        j = j.replace('\n','').lower()
        stop_word_arr.append(j)


# def sentence_segment(text):
#     sents = re.split("([.?!])?[\n]+|[.?!] ", text)
#     return sents


new_sen = []    
for i in sentences:
    i = i.replace('\n','').strip().lower()
    # xóa các ký tự không cần thiết
    # i = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]',' ',i)
    if i.strip() == '' or i.strip() =='.' or i.strip() =='...' or i.strip() =='?' or i.strip() =='??' or i.strip() =='love mom':
        continue
    else:
        i = i.split('\t')
        new_sen.append(i)


for line in new_sen:
    array_temp = []
    string_temp = ''
    if len(line) == 2:
        tokenize_sent = word_tokenize(line[0])
        for word in tokenize_sent:
            if word not in stop_word_arr and word != ',' and word != '.' and word != '/':
                string_temp += (' ' + word)
        array_temp.append(string_temp.strip())
        array_temp.append(line[1])
        dict.append(array_temp)

text = []
label = []
 
for y in dict:
    text.append(y[0])
    label.append(y[1])

# tỉ lệ tập test - train là 8.5 : 1.5
test_percent = 0.15
 
X_train, X_test, y_train, y_test = train_test_split(text, label, test_size=test_percent, random_state=42)
X_train_len = len(X_train)
X_test_len = len(X_test)
y_train_len = len(y_train)
y_test_len = len(y_test)
getTrain = []
getTest = []
 
for x, y in zip(X_train, y_train):
    getTrain.append([x,y])

for x, y in zip(X_test, y_test):
    getTest.append([x,y])


# # encode label
# label_encoder = LabelEncoder()
# label_encoder.fit(y_train)
# print(list(label_encoder.classes_), '\n')
# y_train = label_encoder.transform(y_train)
# y_test = label_encoder.transform(y_test)
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

tf_vectorizer = TfidfVectorizer(ngram_range=(1,2),max_df=0.8,max_features=15000, encoding='utf-8')
X_train_tf_vectorizer = tf_vectorizer.fit_transform(X_train)
X_test_tf_vectorizer = tf_vectorizer.transform(X_test)


MODEL_PATH='C:\\Users\\Admin\\OneDrive\\Desktop\\AD2_LACuong'

# # Logistic Regression
# start_time_lr = time.time()
# LG_model = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=10000)
# LG_model.fit(X_train_tf_vectorizer, y_train)
# lr_time = time.time() - start_time_lr
# y_predict = LG_model.predict(X_test_tf_vectorizer)
# lr_acc = accuracy_score(y_test,y_predict)
# # Save model
# pickle.dump(LG_model, open(os.path.join(MODEL_PATH, "linear_classifier.pkl"), 'wb'))

# # SVM
# start_time_svm = time.time()
# SVM_model = SVC(gamma='scale')
# SVM_model.fit(X_train_tf_vectorizer, y_train)
# svm_time = time.time() - start_time_svm
# y_predict = SVM_model.predict(X_test_tf_vectorizer)
# svm_acc = accuracy_score(y_test,y_predict)
# pickle.dump(SVM_model, open(os.path.join(MODEL_PATH, "svm.pkl"), 'wb'))

start_time = time.time()
text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,2), max_df=0.8, max_features=None)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=10000))
                    ])
LG_model = text_clf.fit(X_train, y_train)
lr_time = time.time() - start_time
y_predict = LG_model.predict(X_test)
lr_acc = accuracy_score(y_test,y_predict)
# Save model
pickle.dump(LG_model, open(os.path.join(MODEL_PATH, "linear_classifier.pkl"), 'wb'))


 
start_time = time.time()
text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,2), max_df=0.8, max_features=None)), 
                     ('tfidf', TfidfTransformer()),
                     ('clf', SVC(gamma='scale'))
                    ])
SVM_model = text_clf.fit(X_train, y_train)
y_predict = SVM_model.predict(X_test)
svm_time = time.time() - start_time
svm_acc = accuracy_score(y_test,y_predict)
# Save model
pickle.dump(text_clf, open(os.path.join(MODEL_PATH, "svm.pkl"), 'wb'))

# Navie Bayes
start_time_navie = time.time()
navie_model = ComplementNB()
navie_model.fit(X_train_tf_vectorizer, y_train)
navie_time = time.time() - start_time_navie
y_predict = navie_model.predict(X_test_tf_vectorizer)
navie_acc = accuracy_score(y_test,y_predict)
# Save model
pickle.dump(navie_model, open(os.path.join(MODEL_PATH, "naive_bayes.pkl"), 'wb'))


start_time = time.time()
num_classes= 7
vocab_size = 5000
embedding_dim = 128
max_len = 20
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size) # adding out of vocabulary token
# tokenizer = Tokenizer(num_words=vocab_size, split=' ') # adding out of vocabulary token
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(X_train)
print(sequences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(LSTM(100, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam', metrics=['accuracy'])


# ///////////////////////
# ///////////////////////
# model = Sequential()
# model.add(Embedding(5000, 120, input_length = 20))
# model.add(SpatialDropout1D(0.9))
# model.add(LSTM(176, dropout=0.2, recurrent_dropout=0.2, input_shape=(16,16)))
# model.add(Dense(32, activation='softmax'))
# model.add(Dense(num_classes, activation='softmax'))
# model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
# print(model.summary())

# ///////////////////////
# ///////////////////////
history = model.fit(padded_sequences, y_train, epochs=40, batch_size=128)
lstm_time = time.time() - start_time
# saving model
model.save("lstm_model.h5")
print('Done training LSTM in', lstm_time, 'seconds.')


# saving tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#saving label encoder
with open('label_encoder.pickle', 'wb') as ecn_file:
    pickle.dump(y_train, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)


# LSTM
model = keras.models.load_model('lstm_model.h5')
# load tokenizer object
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

em=[]
for i in X_test:
    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([i]),
                                            truncating='post', maxlen=20))
    get_label=label_encoder.inverse_transform([np.argmax(result)])[0]
    em.append(get_label)
lstm_acc = np.mean(em == label_encoder.inverse_transform(y_test))

print('\n###############################################')
print("SVM time: ",svm_time)
print("Nav time: ",navie_time)
print("Log time: ",lr_time)
print("LSTM time: ",lstm_time)
print('SVM Accuracy score: ',svm_acc)
print('Nav Accuracy score: ',navie_acc)
print('Log Accuracy score: ',lr_acc)
print('LSTM Accuracy score: ',lstm_acc)
print('###############################################')

list_time=[]
list_time.append(navie_time)
list_time.append(lr_time)
list_time.append(svm_time)
list_time.append(lstm_time)

list_accuracy = []
list_accuracy.append(navie_acc)
list_accuracy.append(lr_acc)
list_accuracy.append(svm_acc)
list_accuracy.append(lstm_acc)

with open('time_accuracy.txt', 'w', encoding="utf-8") as fp:
    for x, y in zip(list_time, list_accuracy):
        fp.write('{} {}\n'.format(x, y))


time_accuracy_arr = []
with open("time_accuracy.txt", "r", encoding="UTF-8") as ta:
    time_accuracy = ta.readlines()
    for i in time_accuracy:
        i = i.replace('\n','').strip()
        i = i.split(' ')
        time_accuracy_arr.append(i)


class GetData(Resource):
    def get(self):
        vui = 0
        buon = 0
        so = 0
        giandu = 0
        ngacnhien = 0
        rac = 0
        yeu=0
        for i in getTrain:
            if i[1] == ':)))))': vui+=1
            if i[1] == ':(((((': buon+=1
            if i[1] == ':s':  so+=1
            if i[1] == 'giận dữ': giandu+=1
            if i[1] == 'ngạc nhiên': ngacnhien+=1
            if i[1] == 'rác': rac+=1
            if i[1] == 'yêu': yeu+=1
        # return getTrain
        return jsonify(
            vui=vui, buon=buon, so=so, giandu=giandu, ngacnhien=ngacnhien, rac=rac, yeu=yeu, tong_train=len(getTrain), array_train=getTrain,
            tong_test=len(getTest), array_test=getTest, tong_all=len(dict), time_accuracy= time_accuracy_arr,
            xtrain=X_train_len, ytrain=y_train_len, xtest=X_test_len, ytest=y_test_len
        )


class PredictSentence(Resource):
    def post(self):
        # Xem kết quả trên từng nhãn
        inIn = request.args.get('string')
        # inIn = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]',' ',inIn)
        inIn = inIn.lower()
        print(inIn)
        string_temp = ''
        tokenize_sent = word_tokenize(inIn)
        for word in tokenize_sent:
            if word not in stop_word_arr and word != ',' and word != '.' and word != '/':
                string_temp += (' ' + word)
        string_temp = string_temp.strip()
        print(string_temp)

        inIn_vectorizer = tf_vectorizer.transform([string_temp])
        
        nb_model1 = pickle.load(open(os.path.join(MODEL_PATH,"linear_classifier.pkl"), 'rb'))
        getthing1 = nb_model1.predict([string_temp])

        nb_model2 = pickle.load(open(os.path.join(MODEL_PATH,"naive_bayes.pkl"), 'rb'))
        getthing2 = nb_model2.predict(inIn_vectorizer)

        nb_model3 = pickle.load(open(os.path.join(MODEL_PATH,"svm.pkl"), 'rb'))
        getthing3 = nb_model3.predict([string_temp])



        # load trained model
        model4 = keras.models.load_model('lstm_model.h5')
        # load tokenizer object
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        # parameters
        result = model4.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([string_temp]),
                                            truncating='post', maxlen=20))

        return jsonify(
            linear_classifier=label_encoder.inverse_transform(getthing1)[0],
            naive_bayes=label_encoder.inverse_transform(getthing2)[0],
            svm=label_encoder.inverse_transform(getthing3)[0],
            lstm=label_encoder.inverse_transform([np.argmax(result)])[0],
        )

api.add_resource(GetData, "/", "/home")
api.add_resource(PredictSentence, "/predict")

if __name__ == '__main__':
    app.run(debug=True)