
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
human_dna = pd.read_table("C:/Users/kashyap ankit/Downloads/human_data (2).txt")
human_dna.head()
chimp_dna = pd.read_table("C:/Users/kashyap ankit/Downloads/chimp_data (2).txt")
chimp_dna.head()
dog_dna = pd.read_table("C:/Users/kashyap ankit/Downloads/dog_data (2).txt")
dog_dna.head()
# plotting a pairplot for the human dataset
sns.pairplot(human_dna,height=5)
plt.suptitle('Pairplot of human Dataset',y=1.02)
plt.show()
# plotting a pairplot for the Chimpanzee dataset
sns.pairplot(chimp_dna,height=5)
plt.suptitle('Pairplot of Chimpanzee Dataset',y=1.02)
plt.show()
def Kmers_funct(seq, size=6):
    return [seq[x:x+size].lower() for x in range(len(seq) - size + 1)]

human_dna['words'] = human_dna.apply(lambda x: Kmers_funct(x['sequence']), axis=1)
human_dna = human_dna.drop('sequence', axis=1)
chimp_dna['words'] = chimp_dna.apply(lambda x: Kmers_funct(x['sequence']), axis=1)
chimp_dna = chimp_dna.drop('sequence', axis=1)
dog_dna['words'] = dog_dna.apply(lambda x: Kmers_funct(x['sequence']), axis=1)
dog_dna = dog_dna.drop('sequence', axis=1)
human_texts = list(human_dna['words'])
for item in range(len(human_texts)):
    human_texts[item] = ' '.join(human_texts[item])
#separate labels
y_human = human_dna.iloc[:, 0].values # y_human for human_dna
#Now let's do the same for chimp and dog.
chimp_texts = list(chimp_dna['words'])
for item in range(len(chimp_texts)):
    chimp_texts[item] = ' '.join(chimp_texts[item])
#separate labels
y_chim = chimp_dna.iloc[:, 0].values # y_chim for chimp_dna

dog_texts = list(dog_dna['words'])
for item in range(len(dog_texts)):
    dog_texts[item] = ' '.join(dog_texts[item])
#separate labels
y_dog = dog_dna.iloc[:, 0].values  # y_dog for dog_dna


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range=(4,4))
X = cv.fit_transform(human_texts)
X_chimp = cv.transform(chimp_texts)
X_dog = cv.transform(dog_texts)

pickle.dump(cv, open("cv.pkl", "wb"))
#Let's see what we have... for human we have 4380 genes converted into uniform length feature vectors of 4-gram k-mer (length 6) counts. For chimp and dog we have the expected same number of features with 1682 and 820 genes respectively.

print(X.shape)
print(X_chimp.shape)
print(X_dog.shape)

human_dna['class'].value_counts().sort_index().plot.bar()
chimp_dna['class'].value_counts().sort_index().plot.bar()
dog_dna['class'].value_counts().sort_index().plot.bar()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_human,test_size = 0.20,random_state=42)
pickle.dump(y_test, open("test.pkl", "wb"))

print(X_train.shape)
print(X_test.shape)

"""A multinomial naive Bayes classifier will be created. I previously did some parameter tuning and found the ngram size of 4 (reflected in the Countvectorizer() 
instance) and a model alpha of 0.1 did the best. Just to keep it simple I won't show that code here."""

###Multinomial Naive Bayes Classifier ###
# The alpha parameter was determined by grid search previously
from sklearn.naive_bayes import MultinomialNB
model= MultinomialNB(alpha=0.1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print("Confusion matrix\n")
print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("Human Gene Prediction and Accuracy:-")
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (100*accuracy,100* precision,100* recall, 100*f1))
y_pred_chimp = model.predict(X_chimp)
y_pred_dog = model.predict(X_dog)
pickle.dump(model, open("model.pkl", "wb"))
#Now for the real test. Let's see how our model perfoms on the DNA sequences from other species. First we'll try the Chimpanzee,
#which we would expect to be very similar to human. Then we will try man's (and woman's) best friend, the Dog DNA sequences.
#Make predictions for the Chimp and dog sequences¶
# Predicting the chimp and worm sequences

# performance on chimp genes
#print("Confusion matrix\n")
#print(pd.crosstab(pd.Series(y_chim, name='Actual'), pd.Series(y_pred_chimp, name='Predicted')))
#accuracy, precision, recall, f1 = get_metrics(y_chim, y_pred_chimp)
#print("Chimp Gene Prediction and Accuracy:-")
#print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (100*accuracy,100* precision,100* recall,100* f1))

# Predicting the dog and worm sequences

# performance on dog genes
#print("Confusion matrix\n")
#print(pd.crosstab(pd.Series(y_dog, name='Actual'), pd.Series(y_pred_dog, name='Predicted')))
#accuracy, precision, recall, f1 = get_metrics(y_dog, y_pred_dog)
#print("Dog Gene Prediction and Accuracy:-")
#print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (100*accuracy, 100*precision,100* recall,100* f1))



