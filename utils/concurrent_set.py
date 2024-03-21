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
    d1=[1,2]
    d3 = [1, 2]
    d2=[1,2]
    c.insert(tuple(d1),1)
    c.insert(tuple(d2),2)
    print(c.get(tuple(d3)))



