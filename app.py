from flask import Flask, request, jsonify, json
from flask_restful import Resource, Api
from flask_cors import CORS

from asyncio.windows_events import NULL
from itertools import count
from underthesea import word_tokenize
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import re
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline
import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalAveragePooling1D, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
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

time_accuracy_arr = []
with open("time_accuracy.txt", "r", encoding="UTF-8") as ta:
    time_accuracy = ta.readlines()
    for i in time_accuracy:
        i = i.replace('\n','').strip()
        i = i.split(' ')
        time_accuracy_arr.append(i)

new_sen = []    
for i in sentences:
    i = i.replace('\n','').strip().lower()
    if i.strip() == '' or i.strip() =='.' or i.strip() =='...' or i.strip() =='?' or i.strip() =='??' or i.strip() =='love mom':
        continue
    else:
        i = i.split('\t')
        if i not in new_sen:
            new_sen.append(i)

for j in stop_word:
    j = j.replace('\n','').lower()
    stop_word_arr.append(j)

for line in new_sen:
    array_temp = []
    string_temp = ''
    # if len(line) == 2:
    tokenize_sent = word_tokenize(line[0])
    for word in tokenize_sent:
        if word not in stop_word_arr and word != ',' and word != '.' and word != '/':
            string_temp += (' ' + word)
    array_temp.append(string_temp.strip())
    array_temp.append(line[1])
    dict.append(array_temp)

# tỉ lệ tập test - train là 8 : 2
test_percent = 0.15
 
text = []
label = []
 
for y in dict:
    text.append(y[0])
    label.append(y[1])
 
X_train, X_test, y_train, y_test = train_test_split(text, label, test_size=test_percent, random_state=42)
X_train_len = len(X_train)
X_test_len = len(X_test)
y_train_len = len(y_train)
y_test_len = len(y_test)
getTrain = []
getTest = []
 
# Lưu train/test data
# Giữ nguyên train/test để về sau so sánh các mô hình cho công bằng
with open('train.txt', 'w', encoding="utf-8") as fp:
    for x, y in zip(X_train, y_train):
    # for x, y in zip(text, label):
        fp.write('{} {}\n'.format(y, x))
        getTrain.append([x,y])
 
with open('test.txt', 'w', encoding="utf-8") as fp:
    for x, y in zip(X_test, y_test):
        fp.write('{} {}\n'.format(y, x))
        getTest.append([x,y])

# encode label
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

tf_vectorizer = TfidfVectorizer(ngram_range=(1,4),max_df=0.8,max_features=15000, encoding='utf-8')
X_train_tf_vectorizer = tf_vectorizer.fit_transform(X_train)
X_test_tf_vectorizer = tf_vectorizer.transform(X_test)

MODEL_PATH='C:\\Users\\Admin\\OneDrive\\Desktop\\AD2_LACuong'


# nb_model3 = pickle.load(open(os.path.join(MODEL_PATH,"svm.pkl"), 'rb'))
# y_pred = nb_model.predict(X_test)
# print(classification_report(y_test, y_pred, target_names=list(label_encoder.classes_)))



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