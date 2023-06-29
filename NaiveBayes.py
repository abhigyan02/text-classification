from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

training_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42, download_if_missing=True)

# we just count the word occurrences
count_vector = CountVectorizer()
x_train_counts = count_vector.fit_transform(training_data.data)

# we transform the word occurrences ito tf-idf
# TfidfTransformer is going to transform CountVectorizer into TfidfVectorizer
tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

model = MultinomialNB().fit(x_train_tfidf, training_data.target)

new = ['Religion and the church provide individuals with a sense of spiritual guidance one can engage in worship and practice their faith.', 'Software engineering is getting hotter and hotter nowadays', 'Medical science is a dynamic field that encompasses research, diagnosis, and treatment to advance healthcare and improve patient outcomes.']

x_new_counts = count_vector.transform(new)
x_new_tfidf = tfidf_transformer.transform(x_new_counts)

predicted = model.predict(x_new_counts)

for doc, category in zip(new, predicted):
    print('%r ---------> %s' % (doc, training_data.target_names[category]))
