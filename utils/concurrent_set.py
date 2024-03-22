import threading
import uuid
class SingletonMapMeta(type):
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
            return cls._instances[cls]

class SingletonMap(metaclass=SingletonMapMeta):
    def __init__(self):
        self._lock = threading.Lock()
        self._map = {}
        self.id = str(uuid.uuid4())
        print('init singleton_map with id', self.id)

    def insert(self, key, value):
        with self._lock:
            self._map[key] = value

    def get(self, key):
        with self._lock:
            return self._map.get(key)

    def get_map(self):
        with self._lock:
            return dict(self._map)  # 返回字典的副本以避免修改原始数据
    def get_id(self):
        return self.id

# 使用单例的函数
def use_singleton_map(thread_id):
    singleton_map = SingletonMap()
    singleton_map.insert(thread_id, f"Value of thread {thread_id}")
    value = singleton_map.get(thread_id)
    print(f"Thread {thread_id}: {value}")

# 创建并启动多个线程
if __name__ == "__main__":
    threads = []
    for i in range(10):
        t = threading.Thread(target=use_singleton_map, args=(i,))
        threads.append(t)
        t.start()

    # 等待所有线程完成
    for t in threads:
        t.join()

    # 打印最终的map内容
    final_map = SingletonMap().get_map()
    print(f"Final map content: {final_map}")