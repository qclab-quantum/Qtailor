import threading
import time

import networkx as nx

from utils.graph_util import GraphUtil


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class ConcurrentMap(metaclass=Singleton):
    def __init__(self):
        self.map = {}
        self.lock = threading.Lock()
        self.visit = 0
        self.hit =0

    def insert(self, key, value):
        self.map[key] = value

    def get(self, key):
        self.visit += 1
        v = self.map.get(key)
        if v is not None:
            self.hit += 1
        return v
    def __sizeof__(self):
        return len(self.map)

    def __str__(self):
        return str(self.map)

if __name__ == '__main__':
    c = ConcurrentMap()
    g1 = GraphUtil.get_new_graph(3)
    #g1.add_edge(0,1)
    # g2=  GraphUtil.get_new_graph(3)
    # g2.add_edge(0, 1)
    # g3 = GraphUtil.get_new_graph(2)
    # g4 = GraphUtil.get_new_graph(3)
    # g5 = GraphUtil.get_new_graph(4)
    start_time = time.time()
    for i in range(10000):
        adj_list=nx.to_dict_of_lists(g1)
        print(adj_list)
        adj_list_str = ''
        for key, value in adj_list.items():
            adj_list_str += f"{key}:{','.join(map(str, value))} "
        c.insert(adj_list_str,1)
        c.get(adj_list_str)
    print(time.time()-start_time)
    print(c)



