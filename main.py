import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

#Llegir l'arxiu amb totes les frases de tots els idiomes
data = pd.read_csv('Dataset/Text/sentences.csv', delimiter='\t', index_col=0, names=['lang','text'], encoding='utf8')

#Filtrar text per idioma
lang = ['spa', 'deu', 'eng', 'fra']
data = data[data['lang'].isin(lang)]

#Seleccionar 50000 frases de cada idioma
data_trim = pd.DataFrame(columns=['lang','text'])
for i in lang:
    lang_trim = data[data['lang'] == i].sample(50000, random_state = 100)
    #data_trim = data_trim.append(lang_trim)
    data_trim = pd.concat([data_trim, lang_trim], ignore_index=True)

#Crear un set d'aprenentatge, de validació i de test aleatori
data_shuffle = data_trim.sample(frac=1)
train = data_shuffle[0:210000]
valid = data_shuffle[210000:270000]
test = data_shuffle[270000:300000]

def obtenir_trigramas(corpus, n_feat = 200):
    """
    Retorna una llista dels 200 trigrames mes comuns d’una llista d’oracions
    El corpus es una llista de strings, i n_feat un enter
    """
    """
    Ajustem el vectoritzador amb CountVectorizer, una funció que permet transformar text i extreure

    característiques d’ell, de manera que analitzem caràcters de 3 en 3, fins un màxim de 200
    """

    vectorizer = CountVectorizer(analyzer='char',
                            ngram_range=(3, 3)
                            ,max_features=n_feat)
    
    #Creem la matriu x, que es la transformació del corpus amb el vectoritzador
    X = vectorizer.fit_transform(corpus)

    #obtenim els noms de les característiques de la matriu, els trigrames
    feature_names = vectorizer.get_feature_names_out()

    return feature_names


#obtenim els trigrames de cada llengua
features = {}
features_set = set()

for i in lang:
    #Obtenir el corpus filtrat per cada llenguatge
    corpus = train[train.lang == i]['text']

    #Obtenir els 200 mots més frequents trigrams
    trigrams = obtenir_trigramas(corpus)

    #Afegir els trigrames en un diccionari
    features[i] = trigrams 
    features_set.update(trigrams)

#Crear una llista de vocabulari utilitzan feature_set
vocab = dict()
for i, f in enumerate(features_set):
    vocab[f] = i

#Entrenar el Count Vectorizer utilitzan el vocabulari
vectorizer = CountVectorizer(analyzer='char',
                             ngram_range=(3, 3),
                            vocabulary=vocab)

#Crear una matriu per el training set
corpus = train['text']   
X = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names_out()

train_feat = pd.DataFrame(data=X.toarray(), columns=feature_names)

#Escalem la matriu utilitzan unba escala de minims i màxims
train_min = train_feat.min()
train_max = train_feat.max()
train_feat = (train_feat - train_min)/(train_max-train_min)

#Afegim una variable objectiu
train_feat['lang'] = list(train['lang'])

#Crear matriu per el validation set
corpus = valid['text']
X = vectorizer.fit_transform(corpus)
valid_feat = pd.DataFrame(data=X.toarray(), columns=feature_names)
valid_feat = (valid_feat - train_min)/(train_max-train_min)
valid_feat['lang'] = list(valid['lang'])

#Crear matriu per el test set
corpus = test['text']
X = vectorizer.fit_transform(corpus)
test_feat = pd.DataFrame(data=X.toarray(), columns=feature_names)
test_feat = (test_feat - train_min)/(train_max-train_min)
test_feat['lang'] = list(test['lang'])