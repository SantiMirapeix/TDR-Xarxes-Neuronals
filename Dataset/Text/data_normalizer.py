import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix

#Importar fitxer de frases
dir = "Dataset/Textos/spa_sentences.csv"
data = pd.read_csv(dir)

#Posar totes les frases en minuscula
data["Sentence"] = data["Sentence"].str.lower()

#Eliminar simbols
def symbol_remover(sentence):
    symbols = "0123456789~`!@#£€$¢¥§%°^&*()-_+={}[]|/:;'<>,.?¿¡"
    for i in symbols:
            if i in sentence:
                sentence = sentence.replace(i, '') #Eliminar simbol
    return sentence

#Vectoritzar frase utilitzant Bag-of-words(Bow)
vectorizer = CountVectorizer(stop_words=([]))
def vector_sentence(sentence):
    vectorizer.fit(sentence)
    vector = vectorizer.fit_transform(sentence)
    return vector

#Iterar cada frase
for index, value in data["Sentence"].iteritems():
    data.at[index, "Sentence"] = symbol_remover(value) #Actualitzar frase amb simbol eliminat
    data.at[index, "Vector"] = vector_sentence(value.split()) #Actualitzar csv amb frases vectoritzades
    print(data.at[index, "Vector"])

#Print del arxiu
print(data)

#Escriure al fitxer antic les modificacions
data.to_csv(dir, index=False)
