# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
from scipy import stats
import sys
import os
import math
import json
import heapq

from process import get_movie_name_id_dict, get_movie_id_name_dict
from process import DoulistFile, MovieFile
from log_tool import data_process_logger as logger

#VecFile = './models/fasttext_model_0804_09_cbow.vec'
import os
root_dir = os.path.dirname(__file__)
VecFile = root_dir + '/../models/fasttext_model_0804_09_skipgram.vec'

# import imp
# imp.reload(sys)

class minHeap():
    '''
    小顶堆
    '''
    def __init__(self, k):
        self._k = k
        self._heap = []

    def add(self, item):
        if len(self._heap) < self._k:
            self._heap.append(item)
            heapq.heapify(self._heap)
        else:
            if item > self._heap[0]:
                self._heap[0] = item
                heapq.heapify(self._heap)

    def get_min(self):
        if len(self._heap) > 0:
            return self._heap[0]
        else:
            return -2

    def get_all(self):
        return self._heap

def similarity(v1, v2):
    '''
    向量相似
    :param v1:
    :param v2:
    :return:
    '''
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    return np.dot(v1, v2) / n1 / n2

def load_vectors(input_file=VecFile):
    '''
    从向量文件里加载向量

    :param input_file:
    :return:
    '''
    vectors = {}
    with open(input_file) as fopen:
        fopen.readline()
        for line in fopen:
            line_list = line.strip().split()

            # </s>符号吧，其它的不得而知
            if not line_list[0].isdigit():
                continue

            movie_id = int(line_list[0])
            vec = np.array([float(_) for _ in line_list[1:]], dtype=float)
            if not movie_id in vectors:
                vectors[movie_id] = vec

    return vectors

def topk_like(cur_movie_name, k=5, print_log=False):
    global movie_name_id_dict
    global movie_id_name_dict
    global vectors
    min_heap = minHeap(k)
    like_candidates = []
    #logger.debug('vecotrs size=%d' % (len(vectors)))
    #logger.debug('cur_movie_name %s, %s' % (cur_movie_name, type(cur_movie_name)))
    if isinstance(cur_movie_name, str):
        cur_movie_name = cur_movie_name.encode('utf8')

    if cur_movie_name not in movie_name_id_dict:
        #logger.info('%s not in movie_name_id_dict[%d]' % (cur_movie_name, len(movie_name_id_dict)))
        return []

    if cur_movie_name not in movie_name_id_dict:
        return []
    cur_movie_id = movie_name_id_dict[cur_movie_name]
    if cur_movie_id not in vectors:
        return []
    cur_vec = vectors[cur_movie_id]
    if print_log:
        logger.info('[%d]%s top %d likes:' % (cur_movie_id, cur_movie_name, k))
    for movie_id, vec in vectors.items():
        if movie_id == cur_movie_id:
            continue
        sim = similarity(cur_vec, vec)
        if len(like_candidates) < k or sim > min_heap.get_min():
            min_heap.add(sim)
            like_candidates.append((movie_id, sim))
    if print_log:
        for t in sorted(like_candidates, reverse=True, key=lambda _:_[1])[:k]:
            logger.info('[%d]%s %f' % (t[0], movie_id_name_dict[t[0]], t[1]))
    return sorted(like_candidates, reverse=True, key=lambda _:_[1])[:k]

def generate_movie_topk_like_result(out_file, movie_file=MovieFile, k=10):
    global movie_id_name_dict
    global movie_name_id_dict
    with open(movie_file) as fopen, open(out_file, 'w') as fwrite:
        for line in fopen:
            line_dict = json.loads(line.strip())
            movie_name = line_dict['movie_name'].encode('utf8')
            if movie_name not in movie_name_id_dict:
                continue
            vector_id = movie_name_id_dict[movie_name]
            if vector_id not in vectors:
                continue
            movie_id = int(line_dict['movie_id'])
            movie_url = 'https://movie.douban.com/subject/%d/' % (movie_id)
            movie_topk_like_tuple = topk_like(movie_name, k)
            fasttext_skipgram_vec = vectors[vector_id].tolist()
            if len(movie_topk_like_tuple) <= 0:
                continue
            movie_like = ["%s\t%f"%(movie_id_name_dict[_[0]], _[1]) for _ in movie_topk_like_tuple]
            movie_dict = {
                    'objectId': movie_id, 
                    'movie_url': movie_url, 
                    'movie_like_fasttext': movie_like,
                    'movie_name': movie_name,
                    'fasttext_skipgram_vec': fasttext_skipgram_vec,
                    }
            fwrite.write(json.dumps(movie_dict)+'\n')


movie_name_id_dict = get_movie_name_id_dict(DoulistFile)
movie_id_name_dict = get_movie_id_name_dict(DoulistFile)
vectors = load_vectors(VecFile)

if __name__ == '__main__':
    # movie_names = ['小时代', '倩女幽魂', '悟空传', '美国往事', '战狼2']
    # for movie_name in movie_names:
    #     topk_like(movie_name, print_log=True)
    #generate_movie_topk_like_result('./output/leancloud_movie_fasttext.json', k=25)

    print(movie_id_name_dict[9006], movie_id_name_dict[7008])
    # 亚洲丰富的饮食世界、面条之路
    # 都是记录片
    # 都是饮食

    l = [19236,14980,7910]
    # 喜剧、gay

    l = [9301, 10232,5879,20025]
    # 铁马寻桥 中国香港 / 剧情 / A Fistful of Stances
    # 900重案追凶 中国香港 / 剧情 / 悬疑 / 惊悚 / 犯罪 / Outburst / 45
    # 学警出更 中国香港 / 犯罪 / 剧情 / 斗气冤家好兄弟 / 45
    # 挞出爱火花 中国香港 / AimingHigh / 挞出爱火花

    l = [18023,3869,520,1244]
    # 女拳霸 泰国 / 动作 / 剧情 / Chocolate / 致命巧克力 / 110分钟
    # 爱久弥新 泰国 / 剧情 / 爱情 / 爱比记忆更长久 / 爱在黃昏 / 117分钟
    # 小情人 泰国 / 儿童 / 喜剧 / My Girl / Fan chan / 110分钟
    # 亲爱的伽利略 泰国 / 剧情 / 爱情 / 与伽利略一起逃亡 / Dear Galileo / 130分钟

    for id in l:
        print(movie_id_name_dict[id])