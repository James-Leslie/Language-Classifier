Naive Bayes classifier trained to discriminate between English, Afrikaans, and Dutch phrases. The classifier was trainined and tested using a labelled dataset containing 2761 phrases. Training was done on a subset of 2000 phrases, while the remaining 761 phrases were kept separate for evaluation of the classifier's accuracy. The final accuracy achieved by the model was 97.5%.

### External Libraries
```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
```

### Data Cleaning and Analysis
First step is to read in the csv file as a pandas dataframe and drop blank lines:
```python
# read input file
phrases = pd.read_csv('lang_data.csv')
# remove empty cells
phrases = phrases.dropna()
```

Clean data further by removing punctuation and capital letters, since these are not useful in distinguishing one language from another:
```python
for phrase in phrases:
  # make lowercase
  phrase[0] = phrase[0].lower()
  # replace punctuation with whitespace
  phrase[0] = phrase[0].translate(str.maketrans('.,:;!?-/%', ' '*len('.,:;!?-/%')))
  # remove extra whitespace
  phrase[0] = " ".join(phrase[0].split())
```

### Feature Extraction
Features are extracted using the Bag of Words representation. In this case, the features are the individual words. The Count Vectorizer creates a matrix which represents the total number of unique words in the corpus of phrases, along with the number of times each word appears.

```python
# create training set from first 2000 phrases
train = phrases[:2000, 0]
targets = phrases[:2000, 1]

# create the bag of words using the count vectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(train)
```

## Training and Testing
The training of the Naive Bayes classifier is done with one line of code:
```python
clf = MultinomialNB().fit(X, targets)
```

Testing is done as follows:
```python
# create test set using all phrases not used in training
test = phrases[2000:, 0]
test_targets = phrases[2000:, 1]

# convert training phrases into bag of words representation
X_new = vectorizer.fit_transform(test)

# classify using the trained model
predicted = clf.predict(X_new)
```

To view accuracy:
```python
test_acc = np.mean(predicted == test_targets)
```
