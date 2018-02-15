from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np


# read input file
phrases = pd.read_csv('lang_data.csv')

# remove empty cells
phrases = phrases.dropna()

# phrase counts by language
eng_count = len(phrases[phrases['language'] == "English"])
afr_count = len(phrases[phrases['language'] == "Afrikaans"])
ned_count = len(phrases[phrases['language'] == "Nederlands"])

# --------------------------------------- Clean data ------------------------------
# Create a 2D numpy array consisting of phrases and classes
phrases = phrases[['text', 'language']].values

for phrase in phrases:

    # make lowercase
    phrase[0] = phrase[0].lower()

    # remove punctuation
    phrase[0] = phrase[0].translate(str.maketrans('.,:;!?-/%', ' '*len('.,:;!?-/%')))

    # remove extra whitespace
    phrase[0] = " ".join(phrase[0].split())


# --------------------------------------- Clean data ------------------------------
# create training set from first 2000 phrases, use remainder for testing
train = phrases[:2000, 0]
targets = phrases[:2000, 1]

test = phrases[2000:, 0]
test_targets = phrases[2000:, 1]

# ------------------------------------- Extract features --------------------------
# create the bag of words using the count vectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(train)

# ------------------------------------- Train and test classifier --------------------------
# train
clf = MultinomialNB().fit(X, targets)

# test
X_new = vectorizer.transform(test)

predicted = clf.predict(X_new)

# compare classified labels to true labels
print("OVERALL TEST ACCURACY:")
print(np.mean(predicted == test_targets))
print("TRAINING SET SIZE: " + str(len(train)))
print("TESTING SET SIZE: " + str(len(test)))
print()

# split predictions into three languages
eng_targs = []
eng_labls = []
afr_targs = []
afr_labls = []
ned_targs = []
ned_labls = []

for i in range(len(test_targets)):
    if test_targets[i] == 'English':
        eng_targs.append(test_targets[i])
        eng_labls.append(predicted[i])

    elif test_targets[i] == "Afrikaans":
        afr_targs.append(test_targets[i])
        afr_labls.append(predicted[i])

    else:
        ned_targs.append(test_targets[i])
        ned_labls.append(predicted[i])

# convert to numpy arrays
eng_targs = np.array(eng_targs)
eng_labls = np.array(eng_labls)
afr_targs = np.array(afr_targs)
afr_labls = np.array(afr_labls)
ned_targs = np.array(ned_targs)
ned_labls = np.array(ned_labls)

# look at individual languages
print("ENGLISH TEST ACCURACY:")
print(np.mean(eng_labls == eng_targs))
print("TRAINING SET SIZE: " + str(eng_count - len(eng_targs)))
print("TESTING SET SIZE: " + str(len(eng_targs)))
print()

print("AFRIKAANS TEST ACCURACY:")
print(np.mean(afr_labls == afr_targs))
print("TRAINING SET SIZE: " + str(afr_count - len(afr_targs)))
print("TESTING SET SIZE: " + str(len(afr_targs)))
print()

print("DUTCH TEST ACCURACY:")
print(np.mean(ned_labls == ned_targs))
print("TRAINING SET SIZE: " + str(ned_count - len(ned_targs)))
print("TESTING SET SIZE: " + str(len(ned_targs)))
