import json


import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import random
import numpy as np
# nltk.download('punkt')

intents = json.loads(
    open(
    "train_data.json"
    ).read()
)



print(intents)

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

lemmatizer = WordNetLemmatizer()


print(words)
# ['How', 'to', 'jump', 'to', 'the', 'specified', 'work', 'station', '?']
words = [
    lemmatizer.lemmatize(word=word) for word in words if word not in ignore_letters
]
print(words)
# ['How', 'to', 'jump', 'to', 'the', 'specified', 'work', 'station']
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))


training = []
output_empty = [0] * len(classes)

# print(documents[0])
# (['How', 'to', 'jump', 'to', 'the', 'specified', 'work', 'station', '?'], 'work station')
for document in documents:
    bag = []
    
    word_patterns = [
        lemmatizer.lemmatize(w.lower()) for w in document[0]
    ]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
print(training)
# training = np.array(training)

train_x = list(map(lambda row: row[0], training))
train_y = list(map(lambda row: row[1], training))
# train_y = list(training[:, 1])


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD



model = Sequential()


exit()