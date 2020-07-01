import pandas as pd
import numpy as np
import re
import nltk #pip install nltk



corpus = ['The sky is blue and beautiful.',
          'Love this blue and beautiful sky!',
          'The quick brown fox jumps over the lazy dog.',
          'The brown fox is quick and the blue dog is lazy!',
          'The sky is very blue and the sky is very beautiful today',
          'The dog is lazy but the brown fox is quick!'
]

labels = ['weather', 'weather', 'animals', 'animals', 'weather', 'animals']

# 第一步：构建DataFrame格式数据
corpus = np.array(corpus)
corpus_df = pd.DataFrame({'Document': corpus, 'categoray': labels})

# 第二步：构建函数进行分词和停用词的去除
# 载入英文的停用词表
stopwords = nltk.corpus.stopwords.words('english')
# 建立词分割模型
cut_model = nltk.WordPunctTokenizer()
# 定义分词和停用词去除的函数
def Normalize_corpus(doc):
    # 去除字符串中结尾的标点符号
    doc = re.sub(r'[^a-zA-Z0-9\s]', '', string=doc)
    # 是字符串变小写格式
    doc = doc.lower()
    # 去除字符串两边的空格
    doc = doc.strip()
    # 进行分词操作
    tokens = cut_model.tokenize(doc)
    # 使用停止用词表去除停用词
    doc = [token for token in tokens if token not in stopwords]
    # 将去除停用词后的字符串使用' '连接，为了接下来的词袋模型做准备
    doc = ' '.join(doc)

    return doc

# 第三步：向量化函数和调用函数
# 向量化函数,当输入一个列表时，列表里的数将被一个一个输入，最后返回也是一个个列表的输出
Normalize_corpus = np.vectorize(Normalize_corpus)
# 调用函数进行分词和去除停用词
corpus_norm = Normalize_corpus(corpus)

# 第四步：使用TfidVectorizer进行TF-idf词袋模型的构建
from sklearn.feature_extraction.text import TfidfVectorizer

# print(type(corpus_norm))
# print(corpus_norm)
Tf = TfidfVectorizer(use_idf=True)
Tf.fit(corpus_norm)
vocs = Tf.get_feature_names()
corpus_array = Tf.transform(corpus_norm).toarray()
corpus_norm_df = pd.DataFrame(corpus_array, columns=vocs)
# print(corpus_norm_df.head())
# 第四步：使用TfidVectorizer进行TF-idf词袋模型的构建
from sklearn.feature_extraction.text import TfidfVectorizer

Tf = TfidfVectorizer(use_idf=True)
Tf.fit(corpus_norm)
vocs = Tf.get_feature_names()
corpus_array = Tf.transform(corpus_norm).toarray()
corpus_norm_df = pd.DataFrame(corpus_array, columns=vocs)
# print(corpus_norm_df.head())

# 第五步：使用cosine_similarity构造相关性矩阵
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(corpus_array)
similarity_matrix_df = pd.DataFrame(similarity_matrix)

# 第六步：构造聚类特征，这里主要是对相关性矩阵做的一种聚类，也可以对词袋模型特征做

from sklearn.cluster import KMeans

model = KMeans(n_clusters=2)
model.fit(np.array(similarity_matrix))
print(model.labels_)
corpus_norm_df['k_labels'] = np.array(model.labels_)
print(corpus_norm_df.head())