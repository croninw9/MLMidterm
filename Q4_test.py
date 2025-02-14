"""
PS#2
Q4 (Testing) - A Small Character Level LSTM
Loads a trained LSTM and mapping and generates sentences

"""

from pickle import load
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

seed_text = 'you will probably be prepared to admit that you are not exceptional'
characters = [10,50,100,500]
for n_chars_to_predict in characters:
   seq_length = 50

   # load the model and mapping
   model = load_model('model50.h5')
   mapping = load(open('mapping.pkl', 'rb'))


   # Make predictions
   for k in range(n_chars_to_predict):
       # encode the characters as integers
       encoded = [mapping[char] for char in seed_text]
       # truncate sequences to a fixed length
       encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
       # one hot encode
       encoded = to_categorical(encoded, num_classes=len(mapping))
       # predict character
       yhat = model.predict_classes(encoded, verbose=0)

       # reverse map integer to character
       for char, index in mapping.items():
           if index == yhat:
               break
       seed_text += char

   print(seed_text)
