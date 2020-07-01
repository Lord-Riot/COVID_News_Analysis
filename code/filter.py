# -*- coding: utf-8 -*-
import pymysql
import os
from enum import Enum


class field_of_new_table(Enum):
    uid = 0
    url = 1
    hash = 2
    title = 3
    publish_time = 4
    content = 5


# 获取关键词
def get_keywords():
    with open("related_keyword.txt") as keyword_file:
        keyword_list = keyword_file.readlines()
    length_of_keyword_list = len(keyword_list)
    index_of_keyword_list = 0
    while index_of_keyword_list < length_of_keyword_list:
        keyword_list[index_of_keyword_list]=keyword_list[index_of_keyword_list].replace('\n', '')
        index_of_keyword_list += 1
    return keyword_list


# 判断是否和COVID-19相关
def is_related_to_COVID(title, content):
    keyword_list = get_keywords()
    for keyword in keyword_list:
        if title.find(keyword) != -1:
            # print(title)
            return True
        elif content.find(keyword) != -1:
            # print(content)
            return True
        else:
            return False


# 执行过滤操作函数
def do_filter():
    connection = pymysql.connect(
            # localhost连接的是本地数据库
            host='localhost',
            # mysql数据库的端口号
            port=3306,
            # 数据库的用户名
            user='root',
            # 本地数据库密码
            passwd='chenhongfan',
            # 数据库名
            db='covid_db',
            # 编码格式
            charset='utf8'
        )
    # 光标对象
    travel_cursor = connection.cursor(pymysql.cursors.SSCursor)
    # 获取所有文本记录
    select_sql = "select * from text_storage"
    # 执行查询，获取总的记录数
    num_of_records = travel_cursor.execute(select_sql)
    # 过滤掉的记录

    ruled_out_items = []
    for item in travel_cursor:
        if not is_related_to_COVID(item[field_of_new_table.title.value], item[field_of_new_table.content.value]):
            ruled_out_items.append(item[field_of_new_table.uid.value])
    connection.close()
    return ruled_out_items


# 清除无关内容
def clean_database(ruled_out_items):
    connection = pymysql.connect(
            # localhost连接的是本地数据库
            host='localhost',
            # mysql数据库的端口号
            port=3306,
            # 数据库的用户名
            user='root',
            # 本地数据库密码
            passwd='chenhongfan',
            # 数据库名
            db='covid_db',
            # 编码格式
            charset='utf8'
        )
    cursor = connection.cursor()
    for item in ruled_out_items:
        delete_sql = "delete from text_storage where uid = "+str(item)+";"
        # print(delete_sql)
        cursor.execute(delete_sql)
    connection.commit()
    connection.close()


if __name__ == "__main__":
    # print(do_filter())
    # print(get_keywords())
    ruled_out_items = do_filter()
    clean_database(ruled_out_items)