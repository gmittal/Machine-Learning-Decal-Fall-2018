import spacy
import tensorflow as tf
import tensorflow_hub as hub

embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
nlp = spacy.load('en')
session = tf.Session()
session.run([tf.global_variables_initializer(), tf.tables_initializer()])

def lemmatize(sent):
    return [word.lemma_ for word in sent]
    
def tokenize(sent):
    return lemmatize([word for word in sent if not word.is_stop])

def embedding(sent):
    return session.run(embed([sent]))

def embed_many(sentences):
    return session.run(embed(sentences))
