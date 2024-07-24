import os
import sys
import argparse
import difflib
import pandas as pd
import numpy as np
import json
from openai import OpenAI

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


OPENAI_KEY = 'sk-djen-bot-XTNkkRE3Lu19zs1oQVPoT3BlbkFJg4kYunBXvyrjZtKd8SFL'
TARGET = 'Intimação'

def preprocess(text):
    """Preprocesses the text by removing non-alphabetic characters, stop words, and stemming the words."""
    return text
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    tokens = word_tokenize(text.lower())
    tokens = [stemmer.stem(token) for token in tokens if token.isalpha() and token not in stop_words]

    return ' '.join(tokens)


def calculate_similarity(text1, text2):
    """Calculates the cosine similarity score between two texts using TF-IDF."""
    texts = [preprocess(text1), preprocess(text2)]
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(texts)

    return cosine_similarity(tfidf)[0][1]

    # for i in range(len(df0)):
    #     print(f'{i}/{len(df0)}', 'computing', df0.iloc[i]['Processo'])
    #     for j in range(len(df1)):
    #         score[i, j] = calculate_similarity(df0.iloc[i][TARGET], df1.iloc[j][TARGET])
    # for i in range(len(df0)):
    #     print(score[i])



def process(df0: pd.DataFrame):
    noccur = df0['Processo'].value_counts().to_dict()
    df0['NOCCUR'] = df0['Processo'].map(lambda x: noccur[x])
    df0['SIMILAR'] = -1

    for index in df0.index:
        #######################################################################################
        # processos que aparecem apenas uma vez ja farao parte da saida
        if df0.loc[index, 'NOCCUR'] == 1:
            df0.loc[index, 'SIMILAR'] = index
            continue

        if df0.loc[index, 'SIMILAR'] != -1:
            continue

        proc = df0.loc[index]
        #######################################################################################
        # processos que aparecem mais de uma vez serao agrupados
        # e analisaremos se conteudo da coluna intimação é identico caracter por caracter
        # se sim entao serao agrupados
        df = df0[df0['Processo'] == proc['Processo']]
        for i in df.index:
            if df0.loc[i, 'SIMILAR']!= -1: continue
            original = df.loc[i, TARGET]
            find = False
            for j in df.index:
                if i >= j: continue
                if df0.loc[j, 'SIMILAR']!= -1: continue                
                edited = df.loc[j, TARGET]
                if original == edited:
                    df0.loc[j, 'SIMILAR'] = i
                    find = True
            if find:
                df0.loc[i, 'SIMILAR'] = i

    # print('--------------------------------------------------------')
    # print(len(df0[df0['SIMILAR']>=0]), len(df0[df0['SIMILAR']<0]), len(df0))
  
    for index in df0.index:
        if df0.loc[index, 'SIMILAR'] != -1: continue
        proc = df0.loc[index]
        df = df0[df0['Processo'] == proc['Processo']]
        # print(index, proc['Processo'], df.index)

        d = difflib.Differ()
        differ = np.zeros((len(df), len(df))) - 1
        imprimir = False
        for i, index_i in enumerate(df.index):
            if df0.loc[index_i, 'SIMILAR']!= -1: continue

            original = df.iloc[i][TARGET].split()
            for j, index_j in enumerate(df.index):
                if df0.loc[index_j, 'SIMILAR']!= -1: continue
                if index_j <= index_i: continue

                edited = df.iloc[j][TARGET].split()

                diff = d.compare(original, edited)
                diff1 = list(diff)
                diff2 = len([a for a in diff1 if a.startswith(("+", "-"))])
                differ[i,j] = diff2
                if diff2 > 0:
                    diffo = [a for a in diff1 if a.startswith(("-", " "))]
                    diffe = [a for a in diff1 if a.startswith(("+", " "))]
                    diffo_i = [k for k, a in enumerate(diffo) if a.startswith(("-"))]
                    diffe_i = [k for k, a in enumerate(diffe) if a.startswith(("+"))]
                    if len(diffo_i) > 0 and len(diffe_i) > 0:
                        diffo_i = diffo_i[0]
                        diffe_i = diffe_i[0]
                        # if diff2 <= 20:
                        imprimir = True
                        if '  Intimado(s)/Citado(s):' in diffo[diffo_i-5:diffo_i] and '  Intimado(s)/Citado(s):' in diffe[diffe_i-5:diffe_i]:
                            df0.loc[index_j, 'SIMILAR'] = index_i
                        # else:
                            # print(index_i, index_j, diff2, proc['Processo'])
                            # print(diffo[diffo_i-5:diffo_i+5])
                            # print(diffe[diffe_i-5:diffe_i+5])
            if imprimir:
                df0.loc[index_i, 'SIMILAR'] = index_i


    # print('--------------------------------------------------------')
    # print(len(df0[df0['SIMILAR']>=0]), len(df0[df0['SIMILAR']<0]), len(df0))
    
    for index in df0.index:
        if df0.loc[index, 'SIMILAR'] != -1: continue
        proc = df0.loc[index]
        df = df0[(df0['Processo'] == proc['Processo']) & (df0['SIMILAR'] == -1)]

        if len(df) == 1:
            df0.loc[index, 'SIMILAR'] = index
            continue

        # print(index, proc['Processo'], df.index)
    
    df00 = pd.concat([df0[df0.index.isin(df0['SIMILAR'].tolist()+[-1])], df0[df0['SIMILAR'] == -1]])

    df0.reset_index().to_excel('output_raw.xlsx', index=False)
    df00.reset_index().to_excel('output_filtered.xlsx', index=False)
    return df00, df0


def compare(df0: pd.DataFrame, df1: pd.DataFrame):
    client = OpenAI(
        api_key=OPENAI_KEY
    )

    procs = df1[(df1['Processo'].isin(df0['Processo'].unique().tolist()))&(df1['Jornal'].str.startswith('D J E N'))]
    # print(procs)
    # print(len(procs), len(procs['Processo'].unique().tolist()), len(df0['Processo'].unique().tolist()), len(df1['Processo'].unique().tolist()))

    df1['GPTSIMILAR'] = -1
    for p in procs['Processo'].unique().tolist():
        df00 = df0[df0['Processo'] == p]
        df11 = df1[df1['Processo'] == p]
        print(p, len(df00), len(df11))
        for i in df11.index:
            original = df11.loc[i, TARGET]

            for j in df00.index:
                edited = df00.loc[j, TARGET]
                # print(repr(original[:100]))
                # print('>', repr(edited[:100]))
            
                msg = f'Compare text0="{original}" with text1="{edited}" and return a json object containing only a boolean value if text0 and text1 has same story in field called "same_story", another bollean for same goal in field called "same_goal", another bollean for same litigation into field called "same_litigation", another field litigation-id with process code'

                completion = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        # {"role": "system", "content": "You are bot thats compare content from text0 with others text inside a array, your answer is in json format with litigation-id, if text-story is the same from text0 for each text inside array."},
                        {"role": "user", "content": msg}
                    ]
                )
                ret = json.loads(completion.choices[0].message.content)                
                print(ret)
                if ret['same_story'] and ret['same_goal'] and ret['same_litigation']:
                    df1.loc[i, 'GPTSIMILAR'] = j+2
                    # print(repr(original))
                    # print('>', repr(edited))
                    break

    print(len(df1[df1['GPTSIMILAR']>=0]),'processos encontrados no dia anterior\n', len(df1[df1['GPTSIMILAR']<0]), 'processos novos\ntotal de processos novos', len(df1))
    return df1


if __name__ == '__main__':
    df0 = pd.read_excel('pub200624.xlsx')
    df1 = pd.read_excel('pub210624.xlsx')
    
    df0f, df0p = process(df0)
    df1f, df1p = process(df1)
    print('removendo', len(df0p[df0p['SIMILAR']>=0]), 'processos semelhantes de', len(df0p), '\n', len(df0p[df0p['SIMILAR']<0]), 'processos nao encontraram pares semelhantes')
    print('--')
    print('removendo', len(df1p[df1p['SIMILAR']>=0]), 'processos semelhantes de', len(df1p), '\n', len(df1p[df1p['SIMILAR']<0]), 'processos nao encontraram pares semelhantes')

    df_end = compare(df0f, df1f)
    df_end.to_excel('output_final.xlsx')

    # print(len(df0p[df0p['SIMILAR']>=0]), len(df0p[df0p['SIMILAR']<0]), len(df0p))
    # print(len(df0f))
    # print(set(df0['Processo'].unique().tolist())-set(df0f['Processo'].unique().tolist()))
    # print('----')
    # print(len(df1p[df1p['SIMILAR']>=0]), len(df1p[df1p['SIMILAR']<0]), len(df1p))
    # print(len(df1f))
    # print(set(df1['Processo'].unique().tolist())-set(df1f['Processo'].unique().tolist()))

    # dfprocs1 = df1[df1['Processo'].isin(df0['Processo'])]
    # procs1 = dfprocs1['Processo'].unique().tolist()
    # dfprocs0 = df0[df0['Processo'].isin(procs1)]
    

    # print('processos repetidos')
    # print(procs1, len(procs1))
    # print(dfprocs1)
    # print(dfprocs0)


