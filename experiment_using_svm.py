#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# experiment using svm

## tfidf transformation w/o pipeline :)
## OR IF YOU WANNA USE PIPELINE:
## model2 = make_pipeline(TfidfVectorizer(analyzer = "word", ngram_range=(1,3)), svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto'))
start = time.time()
tf = TfidfVectorizer(analyzer = "word", ngram_range=(1,3))
word_freq = tf.fit_transform(X_train)
end = time.time()
print('Elapsed time: ',end - start)

word_freq_df = pd.DataFrame(word_freq.toarray(), columns=tf.get_feature_names())
word_freq_df.head(10)

## get feature names to array
## disclaimer: memory error since there is (1,3) ngram range (lot of additional words)
word_list_tf = tf.get_feature_names()
count_list_tf = word_freq.toarray().sum(axis=0) 
word_freq_tf = dict(zip(word_list_tf,count_list_tf))
sorted(word_freq_tf.items(), key=lambda x: x[1], reverse=True)

## modeling with svm.SVC
from sklearn import svm
start = time.time()
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(word_freq, y_train)
end = time.time()
print('Elapsed time: ',end - start)

## accuracy score
predictions_SVM = SVM.predict(X_test)
accuracy_score(y_test, predictions_SVM)

