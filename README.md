# chatbot-python

# 1. 建立聊天機器人

- 準備 n 個input example(s) (A 欄位) 綁定一個intent

```json
{
    "tag": "work station",
    "patterns": [
        "How to jump to the specified work station?"
    ],
    "responses": [
        "You need to log in the administrator account, enter the service center background -> Schedule Control -> Schedule Maintenance, enter the `Warranty Item` page,select the corresponding product serial number, pull down the `Maintenance operation`,click `Schedule Maintenance`, and select `Skip to designated site` to skip."
    ]
},
```
- 寫出對應的回答(B 欄位)
- 訓練模型

## 數據預先處理

### Tokenize
```python
['How', 'to', 'jump', 'to', 'the', 'specified', 'work', 'station', '?', 'How', 'to', 'deal', 'with', 'the', 'discrepancy', 'between', 'the', 'received', 'repair', 'products', 'and', 'the', 'warranty', 'application', '?', 'How', 'to', 'modify', 'the', 'content', 'of', 'the', 'label', 'printed', 'in', 'the', 'station', '?', 'How', 'to', 'upgrade', 'Product', 'SN', 'after', 'replacing', 'spare', 'parts', '?', 'How', 'to', 'work', 'with', 'multiple', 'product', 'failures', '?', 'What', 'should', 'I', 'do', 'if', 'the', 'fault', 'that', 'I', 'initially', 'identified', 'is', 'incorrect', '?', 'How', 'to', 'view', 'multiple', 'job', 'processes', 'in', 'a', 'product', 'job', '?', 'Meet', 'a', 'variety', 'of', 'bad', 'how', 'to', 'deal', 'with', '?']
```

### Lemmatize

- Remove punctuations
- Here the `WordNetLemmatizer` is used.
  - nltk.download() is required.

Lemmatization (外行話) is similar to stemming.

For example,
`works`, `worked`, and `works` will be considered the same as word as `work`.


```python
['How', 'to', 'jump', 'to', 'the', 'specified', 'work', 'station', 'How', 'to', 'deal', 'with', 'the', 'discrepancy', 'between', 'the', 'received', 'repair', 'product', 'and', 'the', 'warranty', 'application', 'How', 'to', 'modify', 'the', 'content', 'of', 'the', 'label', 'printed', 'in', 'the', 'station', 'How', 'to', 'upgrade', 'Product', 'SN', 'after', 'replacing', 'spare', 'part', 'How', 'to', 'work', 'with', 'multiple', 'product', 'failure', 'What', 'should', 'I', 'do', 'if', 'the', 'fault', 'that', 'I', 'initially', 'identified', 'is', 'incorrect', 'How', 'to', 'view', 'multiple', 'job', 'process', 'in', 'a', 'product', 'job', 'Meet', 'a', 'variety', 'of', 'bad', 'how', 'to', 'deal', 'with']

```

### Save the training data as pickle files.
- Save the bag of words (features)
- Save the classes (intent)

### Training data example

```python
(
    ['How', 'to', 'jump', 'to', 'the', 'specified', 'work', 'station', '?'], 
    'work station'
)
```

### Numerical repersentaions

Feature x represents whether there exists the particular word in the question among all the words we have collected so far in our dataset.

Target/label/intent y represents which answer is the correct one that was correspond with the question (x).

x 就是問題當中出現了所有字當中的那些(關鍵)字

y 就是其對應答案

```python
[
    [
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0], 
        [0, 0, 1, 0, 0]
    ], 
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0], 
        [0, 1, 0, 0, 0]
    ], 
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1], 
        [0, 0, 1, 0, 0]
    ], 
    [
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], 
        [0, 0, 0, 1, 0]
    ], 
    [
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0], 
        [0, 0, 1, 0, 0]
    ], 
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1], 
        [0, 0, 0, 0, 1]
    ], 
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 1, 0, 0]
    ], 
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0], 
        [1, 0, 0, 0, 0]
    ]
]
```

## 訓練模型 Train the model

```cmd
Epoch 1/200
2/2 [==============================] - 1s 12ms/step - loss: 1.6985 - accuracy: 0.1250
Epoch 2/200
2/2 [==============================] - 0s 7ms/step - loss: 1.6665 - accuracy: 0.2500
Epoch 3/200
2/2 [==============================] - 0s 7ms/step - loss: 1.4808 - accuracy: 0.3750
Epoch 4/200
2/2 [==============================] - 0s 6ms/step - loss: 1.3691 - accuracy: 0.6250
Epoch 5/200
2/2 [==============================] - 0s 6ms/step - loss: 1.2256 - accuracy: 0.5000
Epoch 6/200
2/2 [==============================] - 0s 6ms/step - loss: 1.3778 - accuracy: 0.3750
Epoch 7/200
2/2 [==============================] - 0s 5ms/step - loss: 1.1788 - accuracy: 0.5000
Epoch 8/200
2/2 [==============================] - 0s 5ms/step - loss: 1.0960 - accuracy: 0.5000
Epoch 9/200
2/2 [==============================] - 0s 6ms/step - loss: 0.9761 - accuracy: 0.7500
Epoch 10/200
2/2 [==============================] - 0s 5ms/step - loss: 0.7803 - accuracy: 0.8750
Epoch 11/200
2/2 [==============================] - 0s 8ms/step - loss: 0.8937 - accuracy: 0.7500
Epoch 12/200
2/2 [==============================] - 0s 10ms/step - loss: 0.9362 - accuracy: 0.5000
Epoch 13/200
2/2 [==============================] - 0s 10ms/step - loss: 0.7259 - accuracy: 0.7500
Epoch 14/200
2/2 [==============================] - 0s 9ms/step - loss: 0.6667 - accuracy: 0.8750
Epoch 15/200
2/2 [==============================] - 0s 8ms/step - loss: 0.8676 - accuracy: 0.7500
Epoch 16/200
2/2 [==============================] - 0s 7ms/step - loss: 0.6896 - accuracy: 0.8750
Epoch 17/200
2/2 [==============================] - 0s 14ms/step - loss: 0.6301 - accuracy: 0.8750
Epoch 18/200
2/2 [==============================] - 0s 7ms/step - loss: 0.7263 - accuracy: 0.5000
Epoch 19/200
2/2 [==============================] - 0s 10ms/step - loss: 0.6145 - accuracy: 0.7500
Epoch 20/200
2/2 [==============================] - 0s 10ms/step - loss: 0.5585 - accuracy: 0.7500
Epoch 21/200
2/2 [==============================] - 0s 11ms/step - loss: 0.3136 - accuracy: 1.0000
Epoch 22/200
2/2 [==============================] - 0s 9ms/step - loss: 0.4103 - accuracy: 0.8750
Epoch 23/200
2/2 [==============================] - 0s 12ms/step - loss: 0.3527 - accuracy: 0.8750
Epoch 24/200
2/2 [==============================] - 0s 7ms/step - loss: 0.4495 - accuracy: 0.8750
Epoch 25/200
2/2 [==============================] - 0s 8ms/step - loss: 0.3820 - accuracy: 1.0000
Epoch 26/200
2/2 [==============================] - 0s 10ms/step - loss: 0.3280 - accuracy: 0.8750
Epoch 27/200
2/2 [==============================] - 0s 11ms/step - loss: 0.3130 - accuracy: 1.0000
Epoch 28/200
2/2 [==============================] - 0s 8ms/step - loss: 0.2021 - accuracy: 1.0000
Epoch 29/200
2/2 [==============================] - 0s 10ms/step - loss: 0.2505 - accuracy: 0.8750
Epoch 30/200
2/2 [==============================] - 0s 10ms/step - loss: 0.1745 - accuracy: 1.0000
Epoch 31/200
2/2 [==============================] - 0s 10ms/step - loss: 0.2330 - accuracy: 0.8750
Epoch 32/200
2/2 [==============================] - 0s 9ms/step - loss: 0.1695 - accuracy: 1.0000
Epoch 33/200
2/2 [==============================] - 0s 8ms/step - loss: 0.1244 - accuracy: 1.0000
Epoch 34/200
2/2 [==============================] - 0s 7ms/step - loss: 0.2687 - accuracy: 0.8750
Epoch 35/200
...
Epoch 198/200
2/2 [==============================] - 0s 9ms/step - loss: 0.0034 - accuracy: 1.0000
Epoch 199/200
2/2 [==============================] - 0s 8ms/step - loss: 3.0289e-04 - accuracy: 1.0000
Epoch 200/200
2/2 [==============================] - 0s 6ms/step - loss: 0.0071 - accuracy: 1.0000
```

# 2. 使用聊天機器人

- 準備好word banks跟model
- 使用者輸入與之前訓練的數據不一樣的新input (C 欄位)
- 模型會依照該輸入的相似程度決定哪一個回答最為輸出
  - 每一個回答都有對應的概率

User input

```cmd
How can a working station operate again?
```

Tokenize and Lemmatize

```cmd
['How', 'can', 'a', 'working', 'station', 'operate', 'again', '?']
```

Use previously saved word bank

```cmd
[1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 1 0 0 0 0 0 0 0 0 0]
```

model.predict() gives out the probability of each answer to be the right one.

```cmd
1/1 [==============================] - 0s 121ms/step
[0.02258058 0.033048   0.858919   0.01683547 0.0686169 ]
```

Use the training data to get the argmax() answer.

```cmd
During the operation of service center, the [Work record] office can select the corresponding fault code several times according to the actual situation. After selecting different fault codes, the system will bring out different operation guidance, and operators can operate according to the operation guidance process.
```