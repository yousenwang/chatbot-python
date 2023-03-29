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



# print(intents)

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
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(
    learning_rate=0.01,
    momentum=0.9,
    nesterov=True
)

model.compile(
    loss='categorical_crossentropy',
    optimizer=sgd,
    metrics=['accuracy']
)

fitted_model = model.fit(
    np.array(train_x),
    np.array(train_y),
    epochs=200,
    batch_size=5,
    verbose=1
)


model.save('chatbot_model.h5', fitted_model)