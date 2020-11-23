#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# function to call the model and return the predict
# you can use joblib or pickel to import previous model

def depression(s):
    pred = model.predict([s])
    predprob = model.predict_proba([s])
    if pred[0] == 1:
        return print('Not depressed\nProbability: ', np.max(predprob))
    else:
         return print('Depressed\nProbability: ', np.max(predprob))
        
print(depression('i love you'))
new_text = input('Input some text here: ')
print(depression(new_text))

