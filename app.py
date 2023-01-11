from turtle import st
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
with open('datacmt2.txt', "r", encoding="UTF-8") as f:
    sentences_1 = f.readlines()
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

time_accuracy_arr = []
with open("time_accuracy.txt", "r", encoding="UTF-8") as ta:
    time_accuracy = ta.readlines()
    for i in time_accuracy:
        i = i.replace('\n','').strip()
        i = i.split(' ')
        time_accuracy_arr.append(i)

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
        for word in tokenize_sent:
            string_temp += word
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

MODEL_PATH='./'


class GetData(Resource):
    def get(self):
        hanhphuc = 0
        buonba = 0
        sohai = 0
        phanno = 0
        ngacnhien = 0
        thongtin=0
        for i in getTrain:
            if i[1] == 'hạnh phúc': hanhphuc+=1
            if i[1] == 'buồn bã': buonba+=1
            if i[1] == 'sợ hãi':  sohai+=1
            if i[1] == 'phẫn nộ': phanno+=1
            if i[1] == 'ngạc nhiên': ngacnhien+=1
            if i[1] == 'thông tin': thongtin+=1
        # return getTrain
        return jsonify(
            hanhphuc=hanhphuc, buonba=buonba, sohai=sohai, phanno=phanno, ngacnhien=ngacnhien, thongtin=thongtin, tong_train=len(getTrain), array_train=getTrain,
            tong_test=len(getTest), array_test=getTest, tong_all=len(dict), time_accuracy= time_accuracy_arr,
            xtrain=X_train_len, ytrain=y_train_len, xtest=X_test_len, ytest=y_test_len
        )


class PredictSentence(Resource):
    def post(self):
        # Xem kết quả trên từng nhãn
        inIn = request.args.get('string')
        inIn = inIn.lower()
        inIn = chuan_hoa_dau_cau_tieng_viet(inIn.strip())
        string_temp = ''
        tokenize_sent = word_tokenize(inIn, format="text")
        for word in tokenize_sent:
            string_temp += word
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