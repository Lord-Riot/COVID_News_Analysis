import os
import json
import jieba
import pymysql
import pandas as pd
from tqdm import tqdm


# 读取文件中的内容，并返回
def read_file(path):
    with open(path, "rb") as fp:
        content = fp.read()
    return content


# 将内容保存到指定文件
def save_file(save_path, content):
    with open(save_path, "wb") as fp:
        fp.write(content.encode('utf-8'))


# 获取json文件的全部内容
def get_json_content(path):
    with open(path, "rb") as file:
        file_json = json.load(file)
    return file_json


# 解析json文件，根据关键词判断是否与疫情相关，相关则提取其内容并分词，存储到指定文件夹下
def extract_content_json(json_path, txt_path, keyword_list, stop_word_list):
    cate_list = os.listdir(json_path)

    if not os.path.exists(txt_path):
        os.makedirs(txt_path)

    for dates in tqdm(cate_list):
        file_now = json_path + "\\" + dates
        json_content = get_json_content(file_now)

        for module in json_content:
            keywords = module['meta']['keyword']
            for word in keywords:
                if word in keyword_list:
                    content = module['meta']['content']
                    content = content.replace("\r\n", "")
                    content = content.replace(" ", "")
                    content_seg = jieba.cut(content, cut_all=False, HMM=True)

                    final_content = []
                    for char in content_seg:
                        if char not in stop_word_list and char != '\t':
                            final_content.append(char)

                    title = module['meta']['title'].replace(" ", "_").replace(".", "").replace("\\", "").replace("?", "")
                    title = title.replace(":", "").replace("|", "-").replace("/", "-").replace("*", "")
                    title = title.replace("&quot;", "")

                    # print(title)
                    save_file(txt_path + "\\" + title + '.txt', " ".join(final_content))
                    continue

    print("分词结束！")


# 通过数据库获取新闻内容，网页uid及时间戳，并将内容进行分词，并存储到csv文件中
def extract_content_sql(host, database, user, password, port, csv_path, stop_word_list):
    db_connect = pymysql.connect(host=host, database=database, user=user, password=password, port=port, charset='utf8')
    sql_cmd = "select uid, content, publish_time, url from text_storage;"

    web_data = pd.read_sql(sql_cmd, db_connect)

    content_list = []
    for content in tqdm(web_data['content']):
        after_content = content.replace("\r\n", "").replace(" ", "")
        after_content = jieba.cut(after_content, cut_all=False, HMM=True)

        tmp_content = []
        for char in after_content:
            if char not in stop_word_list and char != '\t' and char != '\u3000':
                tmp_content.append(char)
        content_list.append(tmp_content)
    web_data['content'] = content_list
    web_data.to_csv(csv_path)


# 指定疫情关联词路径，停止词路径，解析json文件路径，存储分词后文件路径，并完成分词处理
if __name__ == '__main__':
    # 疫情新闻关联词文件路径
    keyword_path = 'D:\\报告汇总\\大型程序设计实践\\data\\news\\related_keyword.txt'
    # 未分词txt文件路径
    data_path = 'D:\\报告汇总\\大型程序设计实践\\data\\news\\data'
    # 分词后txt文件路径
    data_txt_path = 'D:\\报告汇总\\大型程序设计实践\\data\\news\\data_txt'
    # 停止词路径
    stop_words_path = 'D:\\报告汇总\\大型程序设计实践\\handle_data\\cn_stopwords.txt'
    # 分词后csv文件路径
    data_csv_path = 'D:\\报告汇总\\大型程序设计实践\\data\\news\\web_data.csv'

    # 初始化MySQL数据库参数
    host = "localhost"
    port = 3306
    user = "root"
    password = "yijianshen0619"
    database = "covid_db"
    keyword_dict = read_file(keyword_path).decode()
    stop_word_dict = read_file(stop_words_path).decode()

    # extract_content_json(data_path, data_txt_path, keyword_dict, stop_word_dict)
    extract_content_sql(host, database, user, password, port, data_csv_path, stop_word_dict)
