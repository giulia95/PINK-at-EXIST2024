"""
Probabilistic approach to identify identity terms from Training data.
This approach is based on Spacy Stanza lemmatization in order to group concepts; apply several preprocessing
techniques such as lower case conversion and special character removal. It considers only words that appear at least
10 times in training data.

Requires Spacy stanza: "pip install spacy-stanza"
TODO: see bug: https://github.com/explosion/spacy-stanza/issues/32
"""

import pandas as pd
import argparse
import string
from collections import Counter
import stanza
import spacy_stanza
import re
import os
from tqdm import tqdm
import preprocessing
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# _______________________________________________UTILS_______________________________________________________
# stopwords redefined to keep potentially sexism/misoginy-related terms
stopwords = ["a", "about", "above", "above", "across", "afterwards", "again", "against",
             "all", "almost", "alone", "along", "already", "also", "although", "always", 
             "am", "among", "amongst","amoungst", "amount", "an", "and", "another", "any", 
             "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
             "around", "as", "at", "back", "be", "became", "because", "become", "becomes",
             "becoming", "been", "beforehand", "behind", "being", "below", "beside", "besides",
             "between", "beyond", "bill", "both", "bottom", "but","by", "call", "can", "cannot",
             "cant", "co", "con", "could", "couldnt", "de", "describe", "detail", "do", "done",
             "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else", "elsewhere",
             "empty", "enough","etc", "even", "ever", "every", "everyone", "everything", "everywhere",
             "except", "few", "fifteen", "fify","fill", "find", "fire", "first", "five", "for",
             "former", "formerly", "forty", "found", "four", "from", "front", "full", "further",
             "get", "give", "go", "had", "has", "hasnt", "have", "hence", "here", "hereafter",
             "hereby", "herein", "hereupon", "how", "however", "hundred", "ie", "if", "in",
             "inc", "indeed", "interest", "into", "is", "keep", "last", "latter", "latterly",
             "least", "less", "ltd", "made", "many", "may", "meanwhile", "might", "mill",
             "more", "moreover", "most", "mostly", "move", "much", "must", "name", "namely", "neither", "never",
             "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", 
             "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or",
             "other", "others", "otherwise", "out", "over", "part", "per", "perhaps", "please",
             "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems",
             "serious", "several", "should", "show", "side", "since", "sincere", "six", "sixty",
             "so", "some", "somehow", "someone", "sometime", "sometimes", "somewhere", "still", 
             "such", "system", "take", "ten", "than", "that", "the", "then", "thence", "there",
             "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "thick", 
             "thin", "third", "this", "those", "though", "three", "through", "throughout", 
             "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", 
             "twenty", "two", "un", "under", "until", "up", "upon", "very", "via",
             "was", "well", "were", "what", "whatever", "when", "whence", "whenever",
             "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever",
             "whether", "which", "while", "whither", "who", "whoever", "whole", "whom",
             "whose", "why", "will", "with", "within", "without", "would", "yet","the",
             "ve", "re", "ll", "10", "11", "18", "oh", "s", "t", "m", "did", "don", "got"]

stopwords_es = ["a","actualmente","acuerdo","adelante","ademas","además","adrede","afirmó",
                "agregó","ahi","ahora","ahí","al","algo","alguna","algunas","alguno","algunos",
                "algún","alli","allí","alrededor","ambos","ampleamos","antano","antaño","ante",
                "anterior","antes","apenas","aproximadamente","aquel","aquella","aquellas","aquello",
                "aquellos","aqui","aquél","aquélla","aquéllas","aquéllos","aquí","arriba",
                "arriba", "abajo","aseguró","asi","así","atras","aun","aunque","ayer","añadió","aún",
                "b","bajo","bastante","bien","breve","buen","buena","buenas","bueno","buenos",
                "c","cada","casi","cerca","cierta","ciertas","cierto","ciertos","cinco","claro",
                "comentó","como","con","conmigo","conocer","conseguimos","conseguir","considera",
                "consideró","consigo","consigue","consiguen","consigues","contigo","contra","cosas",
                "creo","cual","cuales","cualquier","cuando","cuanta","cuantas","cuanto","cuantos",
                "cuatro","cuenta","cuál","cuáles","cuándo","cuánta","cuántas","cuánto","cuántos",
                "cómo","d","da","dado","dan","dar","de","debajo","debe","deben","debido","decir",
                "dejó","del","delante","demasiado","demás","dentro","deprisa","desde","despacio",
                "despues","después","detras","detrás","dia","dias","dice","dicen","dicho","dieron",
                "diferente","diferentes","dijeron","dijo","dio","donde","dos","durante","día","días",
                "dónde","e","ejemplo","ello","embargo","empleais","emplean",
                "emplear","empleas","empleo","en","encima","encuentra","enfrente","enseguida","entonces",
                "entre","era","erais","eramos","eran","eras","eres","es","esa","esas","ese","eso","esos",
                "esta","estaba","estabais","estaban","estabas","estad","estada","estadas","estado",
                "estados","estais","estamos","estan","estando","estar","estaremos","estará","estarán",
                "estarás","estaré","estaréis","estaría","estaríais","estaríamos","estarían","estarías",
                "estas","este","estemos","esto","estos","estoy","estuve","estuviera","estuvierais",
                "estuvieran","estuvieras","estuvieron","estuviese","estuvieseis","estuviesen","estuvieses",
                "estuvimos","estuviste","estuvisteis","estuviéramos","estuviésemos","estuvo","está",
                "estábamos","estáis","están","estás","esté","estéis","estén","estés","ex","excepto",
                "existe","existen","explicó","expresó","f","fin","final","fue","fuera","fuerais","fueran",
                "fueras","fueron","fuese","fueseis","fuesen","fueses","fui","fuimos","fuiste","fuisteis",
                "fuéramos","fuésemos","g","general","gran","grandes","gueno","h","ha","haber","habia",
                "habida","habidas","habido","habidos","habiendo","habla","hablan","habremos","habrá",
                "habrán","habrás","habré","habréis","habría","habríais","habríamos","habrían","habrías",
                "habéis","había","habíais","habíamos","habían","habías","hace","haceis","hacemos","hacen",
                "hacer","hacerlo","haces","hacia","haciendo","hago","han","has","hasta","hay","haya",
                "hayamos","hayan","hayas","hayáis","he","hecho","hemos","hicieron","hizo","horas","hoy",
                "hube","hubiera","hubierais","hubieran","hubieras","hubieron","hubiese","hubieseis",
                "hubiesen","hubieses","hubimos","hubiste","hubisteis","hubiéramos","hubiésemos","hubo",
                "i","igual","incluso","indicó","informo","informó","intenta","intentais","intentamos",
                "intentan","intentar","intentas","intento","ir","j","junto","k","l","lado","largo",
                "las","le","lejos","les","llegó","lleva","llevar","lo","los","luego","lugar","m","mal",
                "manera","manifestó","mas","mayor","me","mediante","medio","mejor","mencionó","menos",
                "menudo","mi","mientras","mis","misma","mismas","mismo",
                "mismos","modo","momento","mucha","muchas","mucho","muchos","muy","más","mí",
                "n","nada","nadie","ni","ninguna","ningunas","ninguno","ningunos",
                "ningún","no","nos",
                "nueva","nuevas","nuevo","nuevos","nunca","o","ocho","os","otra","otras","otro",
                "otros","p","pais","para","parece","parte","partir","pasada","pasado","país",
                "peor","pero","pesar","poca","pocas","poco","pocos","podeis","podemos","poder",
                "podria","podriais","podriamos","podrian","podrias","podrá","podrán","podría",
                "podrían","poner","por","por qué","porque","posible","primer","primera","primero",
                "primeros","principalmente","pronto","propia","propias","propio","propios","proximo",
                "próximo","próximos","pudo","pueda","puede","pueden","puedo","pues","q","qeu","que",
                "quedó","queremos","quien","quienes","quiere","quiza","quizas","quizá","quizás",
                "quién","quiénes","qué","r","raras","realizado","realizar","realizó","repente",
                "respecto","s","sabe","sabeis","sabemos","saben","saber","sabes","sal","salvo","se",
                "sea","seamos","sean","seas","segun","segunda","segundo","según","seis","ser","sera",
                "seremos","será","serán","serás","seré","seréis","sería","seríais","seríamos","serían",
                "serías","seáis","señaló","si","sido","siempre","siendo","siete","sigue","siguiente",
                "sin","sino","sobre","sois","sola","solamente","solas","solo","solos","somos","son",
                "soy","soyos","su","supuesto","sus","sé","sí","sólo",
                "t","tal","tambien","también","tampoco","tan","tanto","tarde","te","temprano","tendremos",
                "tendrá","tendrán","tendrás","tendré","tendréis","tendría","tendríais","tendríamos",
                "tendrían","tendrías","tened","teneis","tenemos","tener","tenga","tengamos","tengan",
                "tengas","tengo","tengáis","tenida","tenidas","tenido","tenidos","teniendo","tenéis",
                "tenía","teníais","teníamos","tenían","tenías","tercera","ti","tiempo","tiene","tienen",
                "tienes","todavia","todavía","total","trabaja","trabajais",
                "trabajamos","trabajan","trabajar","trabajas","trabajo","tras","trata","través","tres",
                "tu","tus","tuve","tuviera","tuvierais","tuvieran","tuvieras","tuvieron","tuviese",
                "tuvieseis","tuviesen","tuvieses","tuvimos","tuviste","tuvisteis","tuviéramos",
                "tuviésemos","tuvo","tú","u","ultimo","usa","usais","usamos","usan","usar","usas","uso","usted",
                "ustedes","v","va","vais","valor","vamos","van","vaya","veces",
                "ver","verdad","verdadera","verdadero","vez","voy",
                "w","x","y","ya","yo","z","éramos",
                "última","últimas","último","últimos"]

def clear_text_lemma(testo, language, nlp):
    """
    Remove punctuation, brings to lowercase, remove special char, apply Stanza lemmatization

    :param testo: text to process
    :return: processed text
    """    

    rev = []

    testo = testo.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    testo = testo.lower()
    testo = re.sub(r'\d+', '', testo)
    testo = re.sub('[^A-Za-z0-9 ]+', '', testo)
    testo = " ".join(testo.split())  # single_spaces

    doc = nlp(testo)
    for token in doc:
        rev.append(token.lemma_)

    if language =='es':
        stopwords = stopwords_es
    for word in list(rev):  # iterating on a copy since removing will mess things up
        if word in stopwords:
            rev.remove(word)
    return rev


def frequent_words(text, n):
    """
    remove from the dictionary words that appears less than n times.
    :param text: text to process
    :param n: minimum occurrences number
    """
    split_it = []
    for row in text:
        split_it.extend(row.split())

    counter = Counter(split_it)

    frequent_words = []
    for x in counter.most_common():
        if x[1] >= n:
            frequent_words.append(x[0])
        #else:
        #    print(x)

    return frequent_words

def words_selection(data, id_col, language, nlp):
    # _____________________________________________Dictionary ________________________________________________
    # dictionary includes words that appear at least 10 times

    dictionary = []
    
    for index, row in tqdm(data.iterrows()):
        data.loc[index, 'clear text'] = str(clear_text_lemma(row[2], language, nlp)).replace("'", '')\
                                    .replace(",", '').replace("[",'').replace("]", '')\
                                    .replace("\"", '')
    dictionary = frequent_words(data['clear text'], 10)

    # Word dataframe: a column for each word in the dictionary, with a boolean value to represent its presence in the meme
    word = pd.DataFrame(columns=['id_EXIST'] + dictionary)

    for index, row in tqdm(data.iterrows()):
        new_line = list(data.loc[index, [ 'id_EXIST']])
        #word = pd.concat([word, pd.DataFrame.from_dict({'meme': [data.loc[index, 'meme']], 'id_EXIST': [data.loc[index, 'id_EXIST']]})], ignore_index=True)
        word.loc[len(word)] = [data.loc[index, 'id_EXIST']]  + [None] * (len(word.columns) - 1)  #pd.DataFrame({'meme': [data.loc[index, 'meme']], 'id_EXIST': [data.loc[index, 'id_EXIST']]})], ignore_index=False)
        #word = word.append({'meme': data.loc[index, 'meme'],'id_EXIST': data.loc[index, 'id_EXIST']}, ignore_index=True)
        for w in row['clear text'].split():
            if w in dictionary:
                #print(w)
                word.loc[word['id_EXIST'] == data.loc[index, 'id_EXIST'], w] = 1

    data.columns.values[3] = "meme_id"
    word = pd.DataFrame(columns=[id_col] + dictionary)

    for index, row in tqdm(data.iterrows()):
        #new_line = list(data.loc[index, [ id_col]])
        word.loc[len(word)] = [data.loc[index, id_col]]  + [None] * (len(word.columns) - 1) 
        for w in row['clear text'].split():
            if w in dictionary:
                #print(w)
                word.loc[word[id_col] == data.loc[index, id_col], w] = 1

    word.to_csv('./IdentityTerms/lemma_presence_stanza.csv', index=False)
    return data, word

def compute_conditional_probabilities(word , label='hard_label', col = ['class_misogynous'], epsilon=5000):
    #select the terms i.e. exclude the id and the label
    #NB: assumes that the id is the first column and the label the las one
    col.extend(word.columns[1:len(word.columns)-1].tolist()) 

    condizionate = pd.DataFrame(columns=col)
    condizionate.loc[0, 'class_misogynous'] = 'misogynous'
    for x in word.columns[1:len(word.columns)-1].tolist():
        if len(word.loc[word[label] == 1, x].value_counts()) == 2:
            condizionate.loc[0, x] = word.loc[word[label] == 1, x].value_counts()[1] / \
                                    word.loc[word[label] == 1, x].shape[0]
        elif 1 in word.loc[word[label] == 1, x].tolist():
            condizionate.loc[0, x] = 1 - (1 / epsilon)
        else:
            condizionate.loc[0, x] = (1 / epsilon)

    condizionate.loc[1, 'class_misogynous'] = '¬misogynous'
    for x in word.columns[1:len(word.columns)-1].tolist():
        if len(word.loc[word[label] == 0, x].value_counts()) == 2:
            condizionate.loc[1, x] = word.loc[word[label] == 0, x].value_counts()[1] / \
                                    word.loc[word[label] == 0, x].shape[0]
        elif 1 in word.loc[word[label] == 0, x].tolist():
            condizionate.loc[1, x] = 1 - (1 / epsilon)
        else:
            condizionate.loc[1, x] = (1 / epsilon)
    return condizionate

def compute_evidences_on_tags(word, condizionate):
    # compute P(M|tags)
    calcolate = pd.DataFrame(columns=['meme', 'eq', 'valore'])

    for index, row in word.iterrows():
        
        tags = []
        eq = 'P(M|'
        for i in range(1, len(word.columns)-1):
            if row[i] == 1:
                tags.append(word.columns[i])
                eq = eq + word.columns[i] + ' '
        eq = eq + ')'
        #print('\n')
        #print(eq)

        # values to be normalized
        value_pos = 0.5
        value_neg = 0.5
        conto = '0.5'

        for x in tags:
            conto = conto + '*' + str(condizionate.loc[0, x])
            value_pos = value_pos * condizionate.loc[0, x]
            value_neg = value_neg * condizionate.loc[1, x]

        # Normalization
        somma = value_pos + value_neg
        value_pos = value_pos / somma
        value_neg = value_neg / somma

        calcolate.loc[len(calcolate)] = [index + 1, eq,  value_pos]
        
        eq = 'P(¬M|'
        for i in tags:
            eq = eq + i + ' '
        eq = eq + ')'

        #print(value_pos)
        #result = value_pos
        #print(eq)
        #print(value_neg)
    return calcolate

def evaluate_tags_removal(word, condizionate):
    # Remove tags P(M|tags-{tag})

    rimozioneTag = pd.DataFrame(columns=['meme', 'tagTolto', 'eq', 'valore'])

    for index, row in word.iterrows():
        #print('\n')
        tags = []

        for i in range(1, len(word.columns)-1):
            if row[i] == 1:
                tags.append(word.columns[i])

        # compute probability without selected tag
        for tag in tags:
            tmp = tags.copy()
            tmp.remove(tag)
            eq = 'P(M|'

            value_pos = 0.5
            value_neg = 0.5
            conto = '0.5'

            # values to normaize
            for x in tmp:
                eq = eq + x + ' '
                conto = conto + '*' + str(condizionate.loc[0, x])
                value_pos = value_pos * condizionate.loc[0, x]
                value_neg = value_neg * condizionate.loc[1, x]

            eq = eq + ')'
            #print(eq)
            #print(conto)

            # Normalization
            somma = value_pos + value_neg
            value_pos = value_pos / somma
            value_neg = value_neg / somma
            #print(value_pos)
            rimozioneTag.loc[len(rimozioneTag)] = [index + 1, tag, eq,  value_pos]

    rimozioneTag = rimozioneTag.reset_index(drop=True)
    return rimozioneTag

def compute_meme_scores(word, contitionate):
    # ________________________________________ Meme scores___________________________________________________
    calcolate =  compute_evidences_on_tags(word, contitionate)
    rimozioneTag = evaluate_tags_removal(word, contitionate)

    # valMeme-value
    rimozioneTag['score'] = 0
    for index, row in rimozioneTag.iterrows():
        rimozioneTag.loc[index, 'score'] =  calcolate.loc[calcolate['meme'] == row.meme, 'valore'].values[0] - row.valore

    # Compute mean per tag and save in dataframe
    scores_df = pd.DataFrame(columns=['word', 'score'])
    for tag in word.columns[1:len(word.columns)-1]:
        media = sum(rimozioneTag.loc[rimozioneTag['tagTolto'] == tag, 'score'].tolist()) / len(
            rimozioneTag.loc[rimozioneTag['tagTolto'] == tag, 'score'].tolist())
        scores_df.loc[len(scores_df)] = [tag, media]
        #scores_df = scores_df.append({'word': tag, 'score': media}, ignore_index=True)

    scores_df = scores_df.sort_values(by=['score'], ascending=False)
    #scores_df = scores_df.sort_values(by=['score'], ascending=False)
    scores_df.to_csv('./IdentityTerms/scores_Lemma_Stanza.csv', index=False)
    return scores_df

def compute_identity_terms(data, language="en", id_label = 'id_EXIST', label = 'labels_task4', nlp=spacy_stanza):
    nlp = spacy_stanza.load_pipeline(language)
    data = data.loc[data['lang']==language,:]
    
    if not os.path.exists('./IdentityTerms/'):
        os.makedirs('./IdentityTerms/')

    # ____________________________________________________Load Data______________________________________________
    data['clear text'] = ''

    #data, word = words_selection(data, id_label, language, nlp)
    word = pd.read_csv('./IdentityTerms/lemma_presence_stanza.csv')
    word = word.fillna(0)

    data['hard_label'] = data[label].apply(lambda x: preprocessing.most_frequent(x)[0])
    data['hard_label'] = data['hard_label'].map({'YES': 1, 'NO': 0})

    word['hard_label'] = list(data['hard_label'])
                              
    contitionate = compute_conditional_probabilities(word)
    scores_df = compute_meme_scores(word, contitionate)

    # _______________________________Score analysis___________________________________
    # Remove words with less than 2 char
    short = []
    for w in scores_df.word:
        if len(w) <= 2:
            short.append(scores_df[scores_df['word'] == w].index[0])
    scores_df = scores_df.drop(index=short)

    # first/last 10 terms
    identity_misogynous = scores_df[0:5].word.tolist()
    identity_non_misogynous = scores_df[scores_df.shape[0] - 5:scores_df.shape[0]].word.tolist()
    identity_non_misogynous.reverse()

    identity_terms = [identity_misogynous, identity_non_misogynous]

    with open('./IdentityTerms/IdentityTerms.txt', 'w') as f:
        f.write(str(identity_terms))
"""
def main(data, input_lang):
    compute_identity_terms(data, input_lang)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_string>")
        sys.exit(1)
    
    df = pd.read_json("../data/EXIST2024/EXIST_2024_Memes_Dataset/training/EXIST2024_training.json", orient='index')
    main(df, sys.argv[1])
"""

def main(data, args):
    stanza.download(args.input_lang)
    compute_identity_terms(data, args.input_lang, args.id_label, args.label)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your script")
    parser.add_argument("input_lang", help="Input language")
    parser.add_argument("--id_label", help="Optional argument 1: id_label", default='id_EXIST')
    parser.add_argument("--label", help="Optional argument 2: label", default='labels_task4')
    args = parser.parse_args()
    df = pd.read_json("../data/EXIST2024/EXIST_2024_Memes_Dataset/training/EXIST2024_training.json", orient='index')
    main(df, args)
