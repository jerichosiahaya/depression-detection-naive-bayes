#cleansing data

import pandas as pd
import string
import nltk

#import dataset
dataset_columns = ["target", "ids", "date", "flag", "user", "text"]
dataset_encode = "ISO-8859-1"
data = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding = dataset_encode, names = dataset_columns)

#hapus kolom lainnya, hanya mengambil kolom target dan text
data.drop(['ids','date','flag','user'],axis = 1,inplace = True)

#cek data null
data['text'].isnull().sum()
#hasilnya 0

#menghilangkan punctuation
def remove_punctuation(text):
    no_punct=[words for words in text if words not in string.punctuation]
    words_wo_punct=''.join(no_punct)
    return words_wo_punct
data['clean_text']=data['text'].apply(lambda x: remove_punctuation(x))
data.head()

#menghilangkan hyperlink
data['clean_text'] = data['clean_text'].str.replace(r"http\S+", "") 
data.head()

#menghilangkan emoji
data['clean_text'] = data['clean_text'].str.replace('[^\w\s#@/:%.,_-]', '', flags=re.UNICODE)

#buat semua kata jadi lowercase
data['clean_text'] = data['clean_text'].str.lower()

#tokenization
nltk.download('punkt')

def tokenize(text):
    split=re.split("\W+",text) 
    return split
data['clean_text_tokenize']=data['clean_text'].apply(lambda x: tokenize(x.lower()))

#stopwords
nltk.download('stopwords')
stopword = nltk.corpus.stopwords.words('english')

def remove_stopwords(text):
    text=[word for word in text if word not in stopword]
    return text
data['clean_text_tokenize_stopwords'] = data['clean_text_tokenize'].apply(lambda x: remove_stopwords(x))
data.head(10)
