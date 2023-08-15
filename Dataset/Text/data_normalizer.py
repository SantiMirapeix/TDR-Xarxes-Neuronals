import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

#Importar fitxer de frases
dir = "Dataset\Textos\sentences.csv"
data = pd.read_csv(dir, names=['lang', 'text'])


#Filtrar text per idioma
lang = ['spa', 'deu', 'eng', 'fra']
data = data[data['lang'].isin(lang)]

#Seleccionar 50000 frases de cada idioma
data_trim = pd.DataFrame(columns=['lang','text'])
for i in lang:
     lang_trim = data[data['lang'] == i].sample(50000, random_state = 100)
     data_trim = data_trim.append(lang_trim)

#Seleccionar aleatoriament el train set, valid set i test set
data_shuffle = data_trim.sample(frac=1)
train = data_shuffle[0:210000]
valid = data_shuffle[210000:270000]
test = data_shuffle[270000:300000]

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
