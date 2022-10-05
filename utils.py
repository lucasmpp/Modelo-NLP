#%%
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from unidecode import unidecode
import sklearn
import keras
from itertools import chain
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
#%%



def prepare_words(word, tipo):
   
    if type(word) is float:
        return ""
    
    else:
        #leave the word lower
        word = word.lower()

        #removing \r\n
        word = re.sub(r"\n|\r"," ",word)

        if tipo == "1":
            #removing numbers
            word = re.sub(r"\d"," ",word)

        #removing a special character 
        word = re.sub(r"\W|_"," ",word)

        #removing words that size is equal a 2 or bigger than 18
        word = re.sub(r"\b\w{0,1}\b|\b\w{17,}\b"," ",word)

        #removing a aditional spaces
        word = re.sub(r"\s{2,}"," ",word)

        #removing acentos
        word = unidecode(word)


        return(word)

#%%
def get_indices(text, palavras_uteis):

    return [key for key, val in enumerate(text) if val in palavras_uteis]


def get_arrange(indice, minimo, maximo):

    if type(indice) is int:
        arrange = range(indice-minimo,indice+maximo-1)

        return list(arrange)

    else:
        arrange = map(lambda x: range(x-minimo, x+maximo), indice)

        return set(chain.from_iterable(arrange))
    

def arrange_text(text, minimo, maximo, palavras_uteis):
    
    text = text.split()

    indices = get_indices(text, palavras_uteis)

    if len(indices) > 0:  

        if type(indices) is int: 
            indices = list(indices)

        else:
            indices = indices

        if len(text) < max(indices) + maximo:
            maximo = len(text) - max(indices)
        
        if min(indices) - minimo < 0:
            minimo = min(indices)

        lista_indices = get_arrange(indices, minimo, maximo)

        lista_text = " ".join(np.array(text)[list(lista_indices)])

        return lista_text
    else:

        return "vazio"


#%%
def arrange_plus(text, minimo, maximo, tipo, palavras_uteis):
    
    if type(text) is float:
        return "vazio"
    else:
        text = text.split("_x000D_\n")

        text = list(map(lambda x: arrange_text(prepare_words(x,tipo),minimo,maximo,palavras_uteis), text))

        text = list(filter(lambda x: x != "vazio", text))
        
    if len(text) == 0:
        return "vazio"
    
    else:
        return " ".join(text)
#%%




import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
#%%

def previsao(x_test,y_test,model):    
    predictions = model.predict(x = x_test, verbose = 0)
    y_pred = [1 * (x[0]>=0.5) for x in predictions]
    confusion_matrix = sklearn.metrics.confusion_matrix(y_test, np.rint(y_pred))
    
    df_cm = pd.DataFrame(confusion_matrix, range(2), range(2))
    # plt.figure(figsize=(10,7))
    sns.set(font_scale=1.4) # for label size
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g') # font size
    plt.show()
    
    score = model.evaluate(x_test, y_test, verbose = 0) 

    print('Test loss:', round(score[0],4)) 
    print('Test accuracy:', round(score[1],4))


    return y_pred, predictions    
#%%    
    
def pre_process(sentences_train, sentences_test, n_vocab, max_len):
    tokenizer = Tokenizer(num_words=n_vocab)
    tokenizer.fit_on_texts(sentences_train)

    x_train = tokenizer.texts_to_sequences(sentences_train)
    x_test = tokenizer.texts_to_sequences(sentences_test)


    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

    #print(sentences_train[4])
    #print(x_train[4])
    
    
    x_train = pad_sequences(x_train, padding='post', maxlen=max_len)
    x_test = pad_sequences(x_test, padding='post', maxlen=max_len)

    #x_train[2,:]
    return vocab_size, x_train, x_test, max_len    
#%%
class Features_data:

    

    def __init__(self, palavras_uteis):
        self.palavras_uteis = palavras_uteis
        pass

    # def prepare_word_class(self, coluna, tipo):
    #     self.data[coluna] = pd.Series(map(lambda x: prepare_words(x, "1"), self.data[coluna])) 

    # def arrange_text_class(self, coluna, minimo, maximo):
    #     self.data[coluna] = pd.Series(map(lambda x: arrange_text(x,minimo,maximo), self.data[coluna]))


    # def separeted_data(self, coluna, filtro):
    #     self.data_modelo = self.data[self.data[coluna] != filtro]
    #     self.data_vazio  = self.data[self.data[coluna] == filtro] 

    # def types_vars(self, colunas_x, colunas_y):
    #     self.x = self.data_modelo[colunas_x]
    #     self.y = self.data_modelo[colunas_y]


    # def train_test(self, test_size, random_state):
    #     from sklearn.model_selection import train_test_split

    #     self.sentences_train, self.sentences_test, self.y_train, self.y_test = (train_test_split(self.x, self.y,test_size = test_size, random_state = random_state))


    # def tokenizer_data(self, n_vocab, max_len):
    #     self.vocab_size, self.x_train, self.x_test, self.maxlen = pre_process(self.sentences_train, self.sentences_test, n_vocab, max_len)
    

    ### Parte de códigos destinados aos bancos de teste:

    

    def data_teste(self, data_test, minimo, maximo):

        palavras_uteis = self.palavras_uteis
        #cleaning the text
        data_test["TEXTO"] = pd.Series(map(lambda x: arrange_plus(x, minimo,maximo,"0", self.palavras_uteis), data_test["TEXTO"]))

        self.linhas = data_test.shape[0]
        #setting all values with probability and FUMO equals 0 
        data_test["FUMO"] = [0 for x in range(self.linhas)]

        data_test["Probabilidade"] = [0 for x in range(self.linhas)]


        #sorting out the results of cleaning text into: 
        #data_test_vazio for values equals to "vazio" and
        #data_test_model for texts with different "vazio" values

        self.data_test_vazio = data_test[data_test["TEXTO"] == "vazio"]

        self.data_test_model = data_test[data_test["TEXTO"] != "vazio"]

        #tokenizing the words
        tokenizer = Tokenizer(num_words = 1000)
        tokenizer.fit_on_texts(self.data_test_model["TEXTO"])

        #creating a vector with values has been tokenized
        words_token = tokenizer.texts_to_sequences(self.data_test_model["TEXTO"])

        #defining the size of vocab
        vocab_size = 1000

        #increasing 0 so that the vectors have equal sizes
        words_token = pad_sequences(words_token, padding='post', maxlen=100)

        self.words_token = words_token
        
    def model(self, model):

        #using the model that has already been trained
        predictions = model.predict(x = self.words_token, verbose = 0)
        y_pred = [1 * (x[0]>=0.5) for x in predictions]

        #creating the columns "Probabilidade" and "FUMO"
        self.data_test_model["Probabilidade"] = predictions
        self.data_test_model["FUMO"] = y_pred

        #join the dataframes "data_test_vazio" and data_test_model
        self.data_final = pd.concat([self.data_test_model, self.data_test_vazio])[["NR_REGISTRO","TEXTO","FUMO","Probabilidade"]]



#%%


