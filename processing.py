import re
import warnings

from sklearn import metrics
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
import os
import pickle
import time
import numpy as np
import csv

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

import torch
import transformers
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer
from joblib import dump

stop_word_arr = []
dict = []
with open('datacmt2.txt', "r", encoding="UTF-8") as f:
    sentences_1 = f.readlines()
# with open('tweet_emotions.txt', "r", encoding="utf-16le", errors='ignore') as f:
#     sentences_2 = f.readlines()
with open('val2.txt', "r", encoding='UTF-8') as f:
    sentences_2 = f.readlines()
with open('info.txt', "r", encoding='UTF-8') as f:
    sentences_3 = f.readlines()
with open('sohai.txt', "r", encoding='UTF-8') as f:
    sentences_4 = f.readlines()

with open("vietnamese-stopwords.txt", "r", encoding="UTF-8") as fi:
    stop_word = fi.readlines()
    for j in stop_word:
        j = j.replace('\n','').lower()
        stop_word_arr.append(j)

uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"
 
def loaddicchar():
    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic
dicchar = loaddicchar()

# Hàm chuyển Unicode dựng sẵn về Unicde tổ hợp (phổ biến hơn)
def convert_unicode(txt):
    return re.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)

bang_nguyen_am = [['a', 'à', 'á', 'ả', 'ã', 'ạ', 'a'],
                  ['ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ', 'aw'],
                  ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'aa'],
                  ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ', 'e'],
                  ['ê', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ee'],
                  ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị', 'i'],
                  ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ', 'o'],
                  ['ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ', 'oo'],
                  ['ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'ow'],
                  ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ', 'u'],
                  ['ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'uw'],
                  ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ', 'y']]
bang_ky_tu_dau = ['', 'f', 's', 'r', 'x', 'j']

nguyen_am_to_ids = {}

for i in range(len(bang_nguyen_am)):
    for j in range(len(bang_nguyen_am[i]) - 1):
        nguyen_am_to_ids[bang_nguyen_am[i][j]] = (i, j)

def chuan_hoa_dau_tu_tieng_viet(word):
    if not is_valid_vietnam_word(word):
        return word

    chars = list(word)
    dau_cau = 0
    nguyen_am_index = []
    qu_or_gi = False
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x == -1:
            continue
        elif x == 9:  # check qu
            if index != 0 and chars[index - 1] == 'q':
                chars[index] = 'u'
                qu_or_gi = True
        elif x == 5:  # check gi
            if index != 0 and chars[index - 1] == 'g':
                chars[index] = 'i'
                qu_or_gi = True
        if y != 0:
            dau_cau = y
            chars[index] = bang_nguyen_am[x][0]
        if not qu_or_gi or index != 1:
            nguyen_am_index.append(index)
    if len(nguyen_am_index) < 2:
        if qu_or_gi:
            if len(chars) == 2:
                x, y = nguyen_am_to_ids.get(chars[1])
                chars[1] = bang_nguyen_am[x][dau_cau]
            else:
                x, y = nguyen_am_to_ids.get(chars[2], (-1, -1))
                if x != -1:
                    chars[2] = bang_nguyen_am[x][dau_cau]
                else:
                    chars[1] = bang_nguyen_am[5][dau_cau] if chars[1] == 'i' else bang_nguyen_am[9][dau_cau]
            return ''.join(chars)
        return word

    for index in nguyen_am_index:
        x, y = nguyen_am_to_ids[chars[index]]
        if x == 4 or x == 8:  # ê, ơ
            chars[index] = bang_nguyen_am[x][dau_cau]
            return ''.join(chars)

    if len(nguyen_am_index) == 2:
        if nguyen_am_index[-1] == len(chars) - 1:
            x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
            chars[nguyen_am_index[0]] = bang_nguyen_am[x][dau_cau]
        else:
            x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
            chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]
    else:
        x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
        chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]
    return ''.join(chars)


def is_valid_vietnam_word(word):
    chars = list(word)
    nguyen_am_index = -1
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x != -1:
            if nguyen_am_index == -1:
                nguyen_am_index = index
            else:
                if index - nguyen_am_index != 1:
                    return False
                nguyen_am_index = index
    return True


def chuan_hoa_dau_cau_tieng_viet(sentence):
    """
        Chuyển câu tiếng việt về chuẩn gõ dấu kiểu cũ.
        :param sentence:
        :return:
        """
    sentence = sentence.lower()
    words = sentence.split()
    for index, word in enumerate(words):
        cw = re.sub(r'(^\\p{P}*)([p{L}.]*\\p{L}+)(\\p{P}*$)', r'\1/\2/\3', word).split('/')
        if len(cw) == 3:
            cw[1] = chuan_hoa_dau_tu_tieng_viet(cw[1])
        words[index] = ''.join(cw)
    return ' '.join(words)

new_sen = []    
for i in sentences_1:
    i = i.replace('\n','').strip().lower()
    if i.strip() == '' or i.strip() =='.' or i.strip() =='...' or i.strip() =='?' or i.strip() =='??' or i.strip() =='love mom':
        continue
    else:
        i = i.split('\t')
        new_sen.append(i)
array_for_label_happy=[]
array_for_label_sad=[]
array_for_label_fury=[]
array_for_label_surprise=[]
array_for_label_scare=[]
array_for_label_info=[]
for i in sentences_2:
    i = i.replace('\n','').strip().lower()
    if i.strip() == '' or i.strip() =='.' or i.strip() =='...' or i.strip() =='?' or i.strip() =='??' or i.strip() =='love mom':
        continue
    if 'http' in i:
        continue
    else:
        i = i.split('\t')
        if len(i) == 2:
            if i[1] == 'hạnh phúc':
                array_for_label_happy.append(i)
            if i[1] == 'buồn bã':
                array_for_label_sad.append(i)
            if i[1] == 'phẫn nộ':
                array_for_label_fury.append(i)
            if i[1] == 'ngạc nhiên':
                array_for_label_surprise.append(i)
            if i[1] == 'sợ hãi':
                array_for_label_scare.append(i)
            if i[1] == 'thông tin':
                array_for_label_info.append(i)
        else:
            print(i)
        # new_sen.append(i)

for i in sentences_3:
    i = i.replace('\n','').strip().lower()
    if i.strip() == '' or i.strip() =='.' or i.strip() =='...' or i.strip() =='?' or i.strip() =='??' or i.strip() =='love mom':
        continue
    if 'http' in i:
        continue
    else:
        i = i.split('\t')
        if len(i) == 2:
            if i[1] == 'thông tin':
                array_for_label_info.append(i)

for i in sentences_4:
    i = i.replace('\n','').strip().lower()
    if i.strip() == '' or i.strip() =='.' or i.strip() =='...' or i.strip() =='?' or i.strip() =='??' or i.strip() =='love mom':
        continue
    if 'http' in i:
        continue
    else:
        i = i.split('\t')
        if len(i) == 2:
            if i[1] == 'sợ hãi':
                array_for_label_scare.append(i)
        else:
            print(i)
        # new_sen.append(i)

get_array_for_label_happy = array_for_label_happy[0]
get_array_for_label_sad = array_for_label_sad
get_array_for_label_fury = array_for_label_fury
get_array_for_label_surprise = array_for_label_surprise
get_array_for_label_scare = array_for_label_scare
get_array_for_label_info = array_for_label_info

new_sen.extend(get_array_for_label_happy)
new_sen.extend(get_array_for_label_sad)
new_sen.extend(get_array_for_label_fury)
new_sen.extend(get_array_for_label_surprise)
new_sen.extend(get_array_for_label_scare)
new_sen.extend(get_array_for_label_info)

print(len(new_sen))
count_happy = 0
count_sad = 0
count_fury = 0
count_surprise = 0
count_scare = 0
count_info = 0
for i in new_sen:
    if i[1] == 'thông tin':
        count_info += 1
    if i[1] == 'hạnh phúc':
        count_happy += 1
    if i[1] == 'phẫn nộ':
        count_fury += 1
    if i[1] == 'ngạc nhiên':
        count_surprise += 1
    if i[1] == 'buồn bã':
        count_sad += 1
    if i[1] == 'sợ hãi':
        count_scare += 1
print('thông tin ' , count_info)
print('hạnh phúc ' , count_happy)
print('buồn bã ' , count_sad)
print('phẫn nộ ' , count_fury)
print('sợ hãi ' , count_scare)
print('ngạc nhiên ' , count_surprise)

for line in new_sen:
    array_temp = []
    string_temp = ''
    if len(line) == 2:
        standard_sent = chuan_hoa_dau_cau_tieng_viet(line[0].strip())
        tokenize_sent = word_tokenize(standard_sent, format="text")
        # tokenize_sent = word_tokenize(line[0].strip(), format="text")
        for word in tokenize_sent:
            string_temp += word
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
getTrain = []
getTest = []
 
for x, y in zip(X_train, y_train):
    getTrain.append([x,y])

for x, y in zip(X_test, y_test):
    getTest.append([x,y])


# Encode label
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

print(label_encoder.classes_)

tf_vectorizer = TfidfVectorizer(ngram_range=(1,2),max_df=0.8, max_features=15000, encoding='utf-8')
X_train_tf_vectorizer = tf_vectorizer.fit_transform(X_train)
X_test_tf_vectorizer = tf_vectorizer.transform(X_test)


MODEL_PATH='C:\\Users\\Admin\\OneDrive\\Desktop\\AD2_LACuong'

# LogisticRegression
start_time = time.time()
text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,2), max_df=0.8)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', LogisticRegression())
                    ])
LG_model = text_clf.fit(X_train, y_train)
lr_time = time.time() - start_time
y_predict = LG_model.predict(X_test)
lr_acc = accuracy_score(y_test,y_predict)
# Save model
pickle.dump(LG_model, open(os.path.join(MODEL_PATH, "linear_classifier.pkl"), 'wb'))


# SVM prediction 
start_time = time.time()
text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,2), max_df=0.8)), 
                     ('tfidf', TfidfTransformer()),
                     ('clf', SVC())
                    ])
SVM_model = text_clf.fit(X_train, y_train)
y_predict = SVM_model.predict(X_test)
svm_time = time.time() - start_time
svm_acc = accuracy_score(y_test,y_predict)
# Save model
pickle.dump(text_clf, open(os.path.join(MODEL_PATH, "svm.pkl"), 'wb'))

# Naive Bayes
start_time_naive = time.time()
naive_model = ComplementNB()
naive_model.fit(X_train_tf_vectorizer, y_train)
naive_time = time.time() - start_time_naive
y_predict = naive_model.predict(X_test_tf_vectorizer)
naive_acc = accuracy_score(y_test,y_predict)
# Save model
pickle.dump(naive_model, open(os.path.join(MODEL_PATH, "naive_bayes.pkl"), 'wb'))


start_time = time.time()
num_classes= 6
vocab_size = 5000
embedding_dim = 128
max_len = 20

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X_train)
# word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(X_train)
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
print("Nav time: ",naive_time)
print("Log time: ",lr_time)
print("LSTM time: ",lstm_time)
print('SVM Accuracy score: ',svm_acc)
print('Nav Accuracy score: ',naive_acc)
print('Log Accuracy score: ',lr_acc)
print('LSTM Accuracy score: ',lstm_acc)
print('Loss value: ',history.history['loss'][-1])
print('Accuracy value: ',history.history['accuracy'][-1])
print('###############################################')


list_time=[]
list_time.append(naive_time)
list_time.append(lr_time)
list_time.append(svm_time)
list_time.append(lstm_time)

list_accuracy = []
list_accuracy.append(naive_acc)
list_accuracy.append(lr_acc)
list_accuracy.append(svm_acc)
list_accuracy.append(lstm_acc)

with open('time_accuracy.txt', 'w', encoding="utf-8") as fp:
    for x, y in zip(list_time, list_accuracy):
        fp.write('{} {}\n'.format(x, y))