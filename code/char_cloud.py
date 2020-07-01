import os
import csv
import pandas as pd
from tqdm import tqdm
from collections import Counter
from pyecharts.charts import WordCloud
from pyecharts.globals import SymbolType


# 读取文件中的内容，并返回
def read_file(path):
    with open(path, "rb") as fp:
        content = fp.read()
    return content


# 将内容保存到指定文件
def save_file(save_path, content):
    with open(save_path, "wb") as fp:
        fp.write(content.encode('utf-8'))


# 计算分词后的词频，并将词频最高的num个词存储到csv文件中
def get_high_frequency(path, csv_path, num, txt=True):
    counter = Counter()

    if txt:
        cate_list = os.listdir(path)
        for txt in tqdm(cate_list):
            content = read_file(path + "\\" + txt).decode().split(' ')
            for char in content:
                if len(char) > 1 and char != '\r\n':
                    counter[char] += 1
    else:
        content_list = pd.read_csv(path)['content']
        for content in tqdm(content_list):
            content = eval(content)
            for char in content:
                if len(char) > 1 and char != '\r\n' and char != '':
                    counter[char] += 1

    index = 1
    csv_file = open(csv_path, 'w', encoding='utf-8')
    for (char, times) in counter.most_common(num):
        csv_file.write(str(index) + ',' + str(char) + ',' + str(times) + '\n')
        index += 1
    print(index)


# 根据给定的词与词频序列，生成词云的html文件
def words_cloud(word_list) -> WordCloud:
    c = (
        WordCloud()
        .add("", word_list, word_size_range=[20, 100], word_gap=5, shape=SymbolType.ROUND_RECT, height="2000", width="2000", emphasis_shadow_color="red")
    )
    return c


if __name__ == '__main__':
    # 存储高频词csv文件路径
    char_csv_path = 'D:\\报告汇总\\大型程序设计实践\\data\\news\\char_times.csv'
    # 分词csv文件路径
    data_csv_path = 'D:\\报告汇总\\大型程序设计实践\\data\\news\\web_data.csv'
    # 分词txt文件路径
    data_txt_path = 'D:\\报告汇总\\大型程序设计实践\\data\\news\\data_txt'
    # 词云图生成路径
    word_cloud_img_path = 'D:\\报告汇总\\大型程序设计实践\\data\\news\\疫情新闻词云图.html'

    get_high_frequency(data_csv_path, char_csv_path, 80, False)

    with open(char_csv_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        words = []
        for item in reader:
            words.append((item[1], item[2]))
    words_cloud(words).render(word_cloud_img_path)
