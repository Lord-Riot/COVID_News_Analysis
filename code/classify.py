import os
import base64
import pymysql
import numpy as np
import pandas as pd
from sklearn.datasets.base import Bunch
import _pickle as pickle
import matplotlib.pyplot as plt
from math import exp
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier


# 读取文件中的内容，并返回
def read_file(path):
    with open(path, "rb") as fp:
        content = fp.read()
    return content


# 从文件读取bunch结构
def _read_bunch_obj(path):
    with open(path, "rb") as file_obj:
        bunch = pickle.load(file_obj)
    return bunch


# 获取bunch结构的TF-IDF数据
def vector_space(bunch, train_path, stop_word_list):
    space = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[], vocabulary={})
    train_bunch = _read_bunch_obj(train_path)
    space.vocabulary = train_bunch.vocabulary

    vectorizer = TfidfVectorizer(stop_words=stop_word_list, sublinear_tf=True, max_df=0.5, vocabulary=train_bunch.vocabulary)
    space.tdm = vectorizer.fit_transform(bunch.contents)

    return space


# 从csv文件或txt文件集中读取分词数据，计算词频，计算TF-IDF矩阵，并使用K_means算法聚类，将结果可视化并保存（默认聚类簇数为5）
def work(path, csv_path, png_path, csv=False, limit=1000, cluster_num=5, generate=True):
    vector = TfidfVectorizer(use_idf=True)
    data_list = []
    uid_list = []
    time_list = []
    url_list = []

    confidence_list = []
    heat_list = []
    number = 0

    # 从csv文件或txt文件中读取分词数据
    if not csv:
        txt_list = os.listdir(path)
        for txt in txt_list:
            total_content = read_file(path + "\\" + txt).decode()
            data_list.append(total_content)
            number += 1
            if number > limit:
                break
    else:
        content_array = pd.read_csv(path)['content']
        uid_array = pd.read_csv(path)['uid']
        time_array = pd.read_csv(path)['publish_time']
        url_array = pd.read_csv(path)['url']

        for (content, uid, time, url) in zip(content_array, uid_array, time_array, url_array):
            content = eval(content)
            content = ' '.join(content)
            data_list.append(content)
            uid_list.append(uid)
            time_list.append(time)
            url_list.append(url)
            number += 1
            if number > limit:
                break

    # 提取特征
    model_TFIDF = vector.fit_transform(data_list)
    model_KMEANS = KMeans(n_clusters=cluster_num)
    model_KMEANS.fit(model_TFIDF)

    classify_labels = model_KMEANS.labels_
    word_vectors = vector.get_feature_names()
    word_values = model_TFIDF.toarray()
    comment_matrix = np.hstack((word_values, classify_labels.reshape(word_values.shape[0], 1)))

    pre = model_KMEANS.fit_predict(word_values)

    # 生成或读取特征数据
    if generate:
        word_vectors.append('classify_labels')
        comment_frame = pd.DataFrame(comment_matrix, columns=word_vectors)
        comment_frame.to_csv(csv_path)
    else:
        comment_frame = pd.read_csv(csv_path)

    comment_cluster = comment_frame[comment_frame['classify_labels'] == 1].drop('classify_labels', axis=1)
    word_importance = np.sum(comment_cluster, axis=0)
    key_words = word_importance.sort_values(ascending=False)[:cluster_num+1]

    labels_list = list(key_words.index[1:])
    if csv:
        value_list = list(key_words.values[1:])
    else:
        value_list = list(key_words.value[1:])

    print(key_words[1:])

    # 根据关键词和时间计算每条新闻热度和可信度
    if csv:
        # 初始化可信度计算模型
        train_path = "D:/报告汇总/大型程序设计实践/handle_data/judge/train_word_bag/space.dat"
        stop_word_path = "D:/报告汇总/大型程序设计实践/handle_data/judge/train_word_bag/cn_stopwords.txt"
        stop_word_list = read_file(stop_word_path).splitlines()
        train_set = _read_bunch_obj(train_path)
        clf = SGDClassifier(loss="log", penalty="l2").fit(train_set.tdm, train_set.label)

        # 根据不同网站的alexa排名，给出初始值
        alexa = {'sina': 1.1, 'souhu': 1.25, 'bilibili': 1.03, 'huanqiu': 1.05, 'baidu': 1.33}

        for (index, time_str, content, url) in zip(pre, time_list, data_list, url_list):
            init_score = value_list[index]
            time = str_to_datetime(time_str)

            heat_list.append(heat_calculate(init_score, time))
            confidence_list.append(confidence_calculate(labels_list[index], content, stop_word_list, clf, train_path, url, alexa))

    # 降维至2维
    pca = PCA(n_components=2)
    new_data = pca.fit_transform(comment_frame)

    # 可视化聚类效果
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    color = ['c', 'b', 'g', 'r', 'm']
    showed = []
    for i in range(len(new_data)):
        if pre[i] in showed:
            plt.scatter(new_data[i][0], new_data[i][1], color=color[pre[i]], s=100)
        else:
            plt.scatter(new_data[i][0], new_data[i][1], color=color[pre[i]], s=100, label=labels_list[pre[i]])
            showed.append(pre[i])
    plt.legend()
    plt.title("疫情新闻聚类-关键词")

    plt.savefig(png_path, format='png', bbox_inches='tight', transparent=True, dpi=600)
    plt.show()

    # 将分析出的结果存入sql
    host = "localhost"
    port = 3306
    user = "root"
    password = "yijianshen0619"
    database = "covid_db"

    db_connect = pymysql.connect(host=host, database=database, user=user, password=password, port=port, charset='utf8')
    cus = db_connect.cursor()
    cus.execute("truncate covid_db.statistics_result")

    sql = "insert into statistics_result(no, time_of_update, category, confidence, hot_spot_degree) values(%s,%s,%s,%s,%s)"

    for (uid, time, category, confidence, heat) in zip(uid_list, time_list, pre, confidence_list, heat_list):
        cus.execute(sql, [int(uid),  time, int(category), float(confidence), float(heat)])

    db_connect.commit()
    cus.close()
    db_connect.close()


# 将png图像转化为base64编码形式的txt
def png_to_base64(path, target_path):
    with open(path, "rb") as f:
        base64_data = base64.b64encode(f.read())
        file = open(target_path, 'wt')
        file.write(base64_data.decode())
        file.close()


# 将字符串格式的日期转化未datatime格式
def str_to_datetime(time_str):
    return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")


# 计算新闻的热度
def heat_calculate(init_score, time):
    cluster_factor = 1000
    time_factor = 0.0001

    cluster_score = init_score * cluster_factor
    time_score = exp(time_factor*(datetime.now()-time).seconds)
    # print(cluster_score, time_score)

    return cluster_score/time_score


# 计算新闻可信度
def confidence_calculate(label, content, stop_word_list, clf, train_path, url, alexa_dict):
    # bunch规格化
    bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    bunch.target_name.append(label)
    bunch.label.append(label)
    bunch.filenames.append(label)
    bunch.contents.append(content)

    # 计算TF-IDF并预测
    bunch = vector_space(bunch, train_path, stop_word_list)
    predicted = clf.predict_proba(bunch.tdm)
    predict_score = predicted[0][0]

    # 根据alexa排名，得到url对应的初始分值
    alexa_score = 0
    for domain in alexa_dict:
        if domain in url:
            alexa_score = alexa_dict[domain]

    # 综合得出可信度
    confidence = predict_score * 0.95 + (alexa_score-1) * 0.05
    return confidence


if __name__ == '__main__':
    # 分词txt文件路径
    data_txt_path = 'D:\\报告汇总\\大型程序设计实践\\data\\news\\data_txt'
    # 分词csv文件路径
    data_csv_path = 'D:\\报告汇总\\大型程序设计实践\\data\\news\\web_data.csv'
    # 停止词路径
    stop_words_path = 'D:\\报告汇总\\大型程序设计实践\\handle_data\\cn_stopwords.txt'
    # 生成聚类图像路径
    cluster_png_path = 'D:\\报告汇总\\大型程序设计实践\\data\\news\\cluster.png'
    # 图片转base64后路径
    base64_path = 'D:\\报告汇总\\大型程序设计实践\\data\\news\\cluster.txt'
    # 中间模型存储csv路径
    model_csv_path = "D:\\报告汇总\\大型程序设计实践\\handle_data\\result.csv"

    stop_words_dict = read_file(stop_words_path).decode()

    work(data_csv_path, model_csv_path, cluster_png_path, csv=True)
    # work(data_txt_path, model_csv_path, cluster_png_path, csv=False)
    png_to_base64(cluster_png_path, base64_path)
