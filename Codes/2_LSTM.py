# Script Text mining with LSTM
# Transfer learning is still under development, so if the result is not good, 
# some variables should be changed or repeat transfer learning proces for more
# times or just remove is with setting "TransferLearning = False"
from __future__ import print_function
import yaml
import sys
import numpy as np
import pandas as pd
import multiprocessing
import string
import pickle
import os
import re
import tqdm
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from gensim.models.doc2vec import LabeledSentence
from gensim import utils
from gensim.models import Doc2Vec
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.layers.core import Dense, Dropout,Activation
from keras.models import model_from_yaml
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import model_from_yaml
from nltk.corpus import stopwords
np.random.seed(1337)
sys.setrecursionlimit(1000000)

TransferLearning = True
#############################################################################################################################
# Input data
#############################################################################################################################
train_variant = pd.read_csv("training_variants")
test_variant = pd.read_csv("test_variants")
train_text = pd.read_csv("training_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
test_text = pd.read_csv("test_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
train = pd.merge(train_variant, train_text, how='left', on='ID')
# Obtain train_y (3689)
train_y = train['Class'].values
label = open("stage1_solution.csv")
lines = label.readlines()[1:]
label = []
for line in lines:
    main = line.strip().split(',')
    main = main[1:10]
    label.append(main.index('1') + 1)

label = np.array(label)
train_y = np.concatenate((train_y, label), axis=0)
# Obtain train_x (3689*4)
train_x = train.drop('Class', axis=1)
test_x = pd.merge(test_variant, test_text, how='left', on='ID')
ID = pd.DataFrame(pd.read_csv("stage1_solution.csv")["ID"])
test_x = pd.merge(test_x, ID, how='right', on='ID')
train_x = np.concatenate((train_x, test_x), axis=0)
train_x = pd.DataFrame(train_x)
train_x.columns = ["ID", "Gene", "Variation", "Text"]
train_size = train_x.shape[0]
train_x_index = range(train_size)
# Obtain stage2_test (986*4)
stage2_test_variant = pd.read_csv("stage2_test_variants.csv")
stage2_test_text    = pd.read_csv("stage2_test_text.csv", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
stage2_test = pd.merge(stage2_test_variant, stage2_test_text, how='left', on='ID')
test_size = stage2_test.shape[0]
test_index = range(test_size)
# Put all data togethere
all_data = np.concatenate((train_x, stage2_test), axis=0)
all_data = pd.DataFrame(all_data)
all_data.columns = ["ID", "Gene", "Variation", "Text"]

#############################################################################################################################
# Text Clean step 1
# (Stopwords from online collection set)
#############################################################################################################################
def removestop(text):
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = text.lower().split()
    stops = ['so', 'his', 't', 'y', 'ours', 'herself', 'your', 'all', 
    'some', 'they', 'i', 'of', 'didn', 
    'them', 'when', 'will', 'that', 'its', 'because', 
    'while', 'those', 'my', 'don', 'again', 'her', 'if',
    'further', 'now', 'does', 'against', 'won', 'same', 
    'a', 'during', 'who', 'here', 'have', 'in', 'being', 
    'it', 'other', 'once', 'itself', 'hers', 'after', 're',
    'just', 'their', 'himself', 'theirs', 'whom', 'then', 'd', 
    'out', 'm', 'mustn', 'where', 'below', 'about', 'isn',
    'shouldn', 'wouldn', 'these', 'me', 'to', 'doesn', 'into',
    'the', 'until', 'she', 'am', 'under', 'how', 'yourself',
    'couldn', 'ma', 'up', 'than', 'from', 'themselves', 'yourselves',
    'off', 'above', 'yours', 'having', 'mightn', 'needn', 'on', 
    'too', 'there', 'an', 'and', 'down', 'ourselves', 'each',
    'hadn', 'ain', 'such', 've', 'did', 'be', 'or', 'aren', 'he', 
    'should', 'for', 'both', 'doing', 'this', 'through', 'do', 'had',
    'own', 'but', 'were', 'over', 'not', 'are', 'few', 'by', 
    'been', 'most', 'no', 'as', 'was', 'what', 's', 'is', 'you', 
    'shan', 'between', 'wasn', 'has', 'more', 'him', 'nor',
    'can', 'why', 'any', 'at', 'myself', 'very', 'with', 'we', 
    'which', 'hasn', 'weren', 'haven', 'our', 'll', 'only',
    'o', 'before']
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    text = text.replace("."," ").replace(","," ")
    return(text)

all_data_clean1 = []
for it in all_data['Text']:
    newT = removestop(it)
    all_data_clean1.append(newT)

#############################################################################################################################
# Text Clean step 2
# Comentropy
#############################################################################################################################
frature_num = 1000
count_vec   = CountVectorizer(max_df=0.95, min_df=1,
                              max_features=frature_num, decode_error='strict',
                              strip_accents='ascii',
                              stop_words='english')

count_value = count_vec.fit_transform(train_x['Text'])
count_feature_names = count_vec.get_feature_names()
count_table = count_value.toarray()
count_frame = pd.DataFrame(count_table)
count_frame.columns = count_feature_names
classes = train_y 
word_count = train_x['Text'].apply(lambda x: len(x.split()))
count_frame["marking"] = classes
count_frame["word_count"] = word_count
group_table = count_frame.groupby(['marking']).agg('sum').rename(columns = lambda x: x + '')
prob = []
for i in range(1, len(group_table)+1):
    ncol = len(group_table.columns)-1
    for j in range(0, ncol):
        prob.append(float(group_table.ix[i,j])/group_table.ix[i,ncol])

prob_table = pd.DataFrame(np.array(prob).reshape(9, frature_num), columns = count_feature_names)     
psum = prob_table.sum()
prob_sum = pd.DataFrame(np.array(psum).reshape(1, frature_num), columns = count_feature_names)
prob_div_probsum = prob_table.div(prob_sum.ix[0], axis='columns')
log_prob = prob_div_probsum.astype('float64').apply(lambda x: np.log(x))
prob_log_prob = prob_table.multiply(log_prob, axis='columns')
comentropy = -prob_log_prob.sum()
comentropy_frame = pd.DataFrame(np.array(comentropy).reshape(frature_num, 1), index = count_feature_names, columns = ['comentropy'])
comentropy_sorted = comentropy_frame.sort_values('comentropy', ascending=1)
gene       = train_x.ix[:,[1,2]]
gene_list1 = list(gene.values.flatten())
gene_list2 = map(str.lower, gene_list1)
gene_list  = list(set(gene_list2))
high_comentropy       = comentropy_sorted.index[675:]
high_comentropy_list1 = [str(x) for x in high_comentropy.tolist()]
high_comentropy_list  = list(set(high_comentropy_list1) - set(gene_list)) 

def removecomentropy(text, stops):
    text = text.split()
    text = [w for w in text if not w in stops]    
    text = " ".join(text)
    return text

all_data_clean2 = []
for it in all_data_clean1:
    newT = removecomentropy(it,high_comentropy_list)
    all_data_clean2.append(newT)

numwtrain = []
for i in range(0, len(all_data_clean2)):
    numwtrain.extend(re.findall(r'[0-9]\w+',all_data_clean2[i]))

add_stop_words = numwtrain
my_stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

#############################################################################################################################
# Text Clean step 3
# remove the stopwords generated
#############################################################################################################################
def constructLabeledSentences(data):
    sentences=[]
    for index, row in data.iteritems():
        sentences.append(LabeledSentence(utils.to_unicode(row).split(), ['Text' + '_%s' % str(index)]))
    return sentences

def textClean(text):
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = text.lower().split()
    # May change here (remove all the stop words)
    stops = set(my_stop_words)
    text = [w for w in text if not w in stops]    
    text = " ".join(text)
    return(text)

def cleanup(text):
    text = textClean(text)
    text= text.translate(string.maketrans("",""))
    return text

all_data_clean2 = pd.DataFrame(all_data_clean2)
all_data_clean2.columns = ["Text"]
allText = all_data_clean2["Text"].apply(cleanup)
sentences = constructLabeledSentences(allText)

#############################################################################################################################
# Use Doc2Vec get the embedding weights
#############################################################################################################################
vocab_dim = 200
n_exposures = 2 # The observation frequency
window_size = 7
cpu_count = 4
maxlen = 100
n_iterations = 5

text_model=None
filename='DOC2vec_model1.pkl'
if os.path.isfile(filename):
    text_model = Doc2Vec.load(filename)
else:
	text_model = Doc2Vec(size=vocab_dim, min_count=n_exposures, window=window_size, workers=cpu_count, iter=n_iterations)
	text_model.build_vocab(sentences)
	text_model.train(sentences, total_examples=text_model.corpus_count, epochs=text_model.iter)
	text_model.save(filename)

gensim_dict = Dictionary()
gensim_dict.doc2bow(text_model.wv.vocab.keys(), allow_update=True)
w2indx = {v: k+1 for k, v in gensim_dict.items()}
w2vec = {word: text_model[word] for word in w2indx.keys()}
data=[]
for sentence in sentences:
    new_txt = []
    for word in sentence[0]:
        try:
            new_txt.append(w2indx[word])
        except:
            new_txt.append(0)
    data.append(new_txt)

sentences_backed = sentences
sentences = data
sentences = sequence.pad_sequences(sentences, maxlen = maxlen)
index_dict = w2indx
word_vectors = w2vec

n_symbols = len(index_dict) + 1
embedding_weights = np.zeros((n_symbols, vocab_dim))
for word, index in index_dict.items():
    embedding_weights[index, :] = word_vectors[word]


text_train_arrays = np.zeros((train_size, maxlen))
text_test_arrays = np.zeros((test_size, maxlen))
for i in range(train_size):
    text_train_arrays[i,] = sentences[i,]

j=0
for i in range(train_size,train_size+test_size):
    text_test_arrays[j] = sentences[i,]
    j=j+1

#############################################################################################################################
# Use Gene Data file (SVD method)
#############################################################################################################################
Gene_INPUT_DIM=25
svd = TruncatedSVD(n_components=25, n_iter=Gene_INPUT_DIM, random_state=12)
one_hot_gene = pd.get_dummies(all_data['Gene'])
truncated_one_hot_gene = svd.fit_transform(one_hot_gene.values)
one_hot_variation = pd.get_dummies(all_data['Variation'])
truncated_one_hot_variation = svd.fit_transform(one_hot_variation.values)

#############################################################################################################################
# Generate train and test
#############################################################################################################################
train_set=np.hstack((truncated_one_hot_gene[:train_size],truncated_one_hot_variation[:train_size],text_train_arrays))
test_set=np.hstack((truncated_one_hot_gene[train_size:],truncated_one_hot_variation[train_size:],text_test_arrays))
x_train = train_set[:3321]
y_train = train_y[:3321]
x_test = train_set[3321:]
y_test = train_y[3321:]
# encode Y 
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
y_train = np_utils.to_categorical((label_encoder.transform(y_train)))
label_encoder = LabelEncoder()
label_encoder.fit(y_test)
y_test = np_utils.to_categorical((label_encoder.transform(y_test)))

#############################################################################################################################
# Bidirectional LSTM Model
#############################################################################################################################
vocab_dim = 200
batch_size = 32
n_epoch = 2
input_length = 150

model = Sequential()
model.add(Embedding(output_dim=vocab_dim, input_dim=n_symbols, mask_zero=True, weights=[embedding_weights], input_length=input_length))
model.add(Bidirectional(LSTM(output_dim=100, init='uniform', inner_init='uniform', activation='tanh', inner_activation='hard_sigmoid',forget_bias_init='one',return_sequences = True),merge_mode='sum'))
model.add(Bidirectional(LSTM(output_dim=100, init='uniform', inner_init='uniform', activation='tanh', inner_activation='hard_sigmoid',forget_bias_init='one',return_sequences = False),merge_mode='sum'))
model.add(Dropout(0.3))
model.add(Dense(100, init='normal', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(80, init='normal', activation='relu'))
model.add(Dense(9, init='normal', activation="softmax"))
ADAM = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=ADAM, metrics=['accuracy'])
estimator = model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=n_epoch,verbose=1, validation_data=(x_test, y_test))
model.save_weights("final_1.model")

##############################################################################################################################
# Transfer learning
#
# The most important variable are n_epoch and the weights for each time which will affect the results a lot!
# This section is still under development but can already get good result
# You can keep this section if you are interested in it or just remove if the result is not so good
##############################################################################################################################
def predict(dataSet):
    y_pred = model.predict_proba(dataSet)
    output_label = []
    for i in y_pred:
        maxvalue = i.max()
        label = list(i).index(maxvalue)
        output_label.append(label+1)
    return output_label

if TransferLearning == True:
    Result = pd.DataFrame()
    n_epoch = 2 
    # 1. Transfer learning for the first time
    test_label = predict(test_set)
    Result["Prediction1"] = test_label
    w = np.array([1]*len(test_set) + [20]*len(train_set)) # 1:20 can be adjusted!
    train_x_new = np.concatenate((test_set, train_set), axis=0)
    train_y_new = np.concatenate((test_label, train_y), axis=0)
    x_train = train_x_new[:4307]
    y_train = train_y_new[:4307]
    x_test = train_x_new[4307:]
    y_test = train_y_new[4307:]
    w = w[:4307]
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train = np_utils.to_categorical((label_encoder.transform(y_train)))
    label_encoder = LabelEncoder()
    label_encoder.fit(y_test)
    y_test = np_utils.to_categorical((label_encoder.transform(y_test)))
    estimator = model.fit(x_train, y_train, sample_weight=w, batch_size=batch_size, nb_epoch=n_epoch,verbose=1, validation_data=(x_test, y_test))
    model.save_weights('final_2.model')
    # 2. Transfer learning for the second time
    test_label = predict(test_set)
    Result["Prediction2"] = test_label
    test_set_copy = test_set[Result["Prediction1"]==Result["Prediction2"]].copy()
    test_label_copy = np.array(test_label)[Result["Prediction1"]==Result["Prediction2"]].copy()
    w = np.array([1]*len(test_set_copy) + [10]*len(train_set)) # 1:10 can be adjusted!
    train_x_new = np.concatenate((test_set_copy, train_set), axis=0)
    train_y_new = np.concatenate((test_label_copy, train_y), axis=0)
    x_train = train_x_new[:(3321+len(test_set_copy))]
    y_train = train_y_new[:(3321+len(test_set_copy))]
    x_test = train_x_new[(3321+len(test_set_copy)):]
    y_test = train_y_new[(3321+len(test_set_copy)):]
    w = w[:(3321+len(test_set_copy))]
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train = np_utils.to_categorical((label_encoder.transform(y_train)))
    label_encoder = LabelEncoder()
    label_encoder.fit(y_test)
    y_test = np_utils.to_categorical((label_encoder.transform(y_test)))
    estimator = model.fit(x_train, y_train, sample_weight=w, batch_size=batch_size, nb_epoch=n_epoch,verbose=1, validation_data=(x_test, y_test))
    model.save_weights('final_3.model')
    # 3. Transfer learning for the third time
    test_label = predict(test_set)
    Result["Prediction3"] = test_label
    test_set_copy = test_set[Result["Prediction1"]==Result["Prediction2"]].copy()
    test_label_copy = np.array(test_label)[Result["Prediction1"]==Result["Prediction2"]].copy()
    Result_copy = Result[Result["Prediction1"]==Result["Prediction2"]].copy()
    test_set_copy = test_set_copy[Result_copy["Prediction3"]==Result_copy["Prediction2"]].copy()
    test_label_copy = test_label_copy[Result_copy["Prediction3"]==Result_copy["Prediction2"]].copy()
    Result_copy = Result_copy[Result_copy["Prediction3"]==Result_copy["Prediction2"]].copy()
    w = np.array([1]*len(test_set_copy) + [5]*len(train_set)) # 1:5 can be adjusted!
    train_x_new = np.concatenate((test_set_copy, train_set), axis=0)
    train_y_new = np.concatenate((test_label_copy, train_y), axis=0)
    x_train = train_x_new[:(3321+len(test_set_copy))]
    y_train = train_y_new[:(3321+len(test_set_copy))]
    x_test = train_x_new[(3321+len(test_set_copy)):]
    y_test = train_y_new[(3321+len(test_set_copy)):]
    w = w[:(3321+len(test_set_copy))]
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train = np_utils.to_categorical((label_encoder.transform(y_train)))
    label_encoder = LabelEncoder()
    label_encoder.fit(y_test)
    y_test = np_utils.to_categorical((label_encoder.transform(y_test)))
    estimator = model.fit(x_train, y_train, sample_weight=w, batch_size=batch_size, nb_epoch=1,verbose=1, validation_data=(x_test, y_test))
    model.save_weights('final_4.model')
    # 4. Transfer learning for the fourth time
    test_label = predict(test_set)
    Result["Prediction4"] = test_label
    test_set_copy = test_set[Result["Prediction1"]==Result["Prediction2"]].copy()
    test_label_copy = np.array(test_label)[Result["Prediction1"]==Result["Prediction2"]].copy()
    Result_copy = Result[Result["Prediction1"]==Result["Prediction2"]].copy()
    test_set_copy = test_set_copy[Result_copy["Prediction3"]==Result_copy["Prediction2"]].copy()
    test_label_copy = test_label_copy[Result_copy["Prediction3"]==Result_copy["Prediction2"]].copy()
    Result_copy = Result_copy[Result_copy["Prediction3"]==Result_copy["Prediction2"]].copy()
    test_set_copy = test_set_copy[Result_copy["Prediction4"]==Result_copy["Prediction2"]].copy()
    test_label_copy = test_label_copy[Result_copy["Prediction4"]==Result_copy["Prediction2"]].copy()
    Result_copy = Result_copy[Result_copy["Prediction4"]==Result_copy["Prediction2"]].copy()
    w = np.array([1]*len(test_set_copy) + [1]*len(train_set)) # 1:1 can be adjusted!
    train_x_new = np.concatenate((test_set_copy, train_set), axis=0)
    train_y_new = np.concatenate((test_label_copy, train_y), axis=0)
    x_train = train_x_new[:(3321+len(test_set_copy))]
    y_train = train_y_new[:(3321+len(test_set_copy))]
    x_test = train_x_new[(3321+len(test_set_copy)):]
    y_test = train_y_new[(3321+len(test_set_copy)):]
    w = w[:(3321+len(test_set_copy))]
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train = np_utils.to_categorical((label_encoder.transform(y_train)))
    label_encoder = LabelEncoder()
    label_encoder.fit(y_test)
    y_test = np_utils.to_categorical((label_encoder.transform(y_test)))
    estimator = model.fit(x_train, y_train, sample_weight=w, batch_size=batch_size, nb_epoch=1,verbose=1, validation_data=(x_test, y_test))
    model.save_weights('final_5.model')

#########################################
# Generate two files for xgboost 
# Text Mining Done!
#########################################
y_pred = model.predict_proba(test_set)
submission = pd.DataFrame(y_pred)
submission['id'] = range(1,987)
submission.columns = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9', 'ID']
submission.to_csv("TestForXGB.csv",index=False)

y_pred = model.predict_proba(train_set)
submission = pd.DataFrame(y_pred)
submission['id'] = train_x_index
submission.columns = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9', 'ID']
submission.to_csv("TrainForXGB.csv",index=False)

