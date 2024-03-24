import redis
from concurrent.futures import ThreadPoolExecutor

class RedisThreadPool:
    _executor = ThreadPoolExecutor(10)  # 假设我们使用10个线程的线程池
    _redis_pool = None

    @classmethod
    def initialize(cls, host='localhost', port=6379, db=0):
        cls._redis_pool = redis.ConnectionPool(host=host, port=port, db=db)

    @classmethod
    def _get_redis_connection(cls):
        if cls._redis_pool is None:
            raise Exception("Redis connection pool is not initialized.")
        return redis.Redis(connection_pool=cls._redis_pool)

    @classmethod
    def insert(cls, key, value):
        def _insert():
            conn = cls._get_redis_connection()
            print(f"key={key} value={value}")
            conn.set(key, value)
        cls._executor.submit(_insert)


    @classmethod
    def get(cls, key):
        def _read():
            conn = cls._get_redis_connection()
            return conn.get(key)
        future = cls._executor.submit(_read)
        return future.result()  # 等待结果并返回

if __name__ == '__main__':

    # 初始化Redis连接池
    RedisThreadPool.initialize()
    arr=[123]
    print(str(arr))
    # 使用类方法
    RedisThreadPool.insert(str(arr), 1)
    RedisThreadPool.update(str(arr), 2)
    value = RedisThreadPool.get(str(123123))
    print(value is None)
    #print(int(value))  # 应该打印出 'new_value'，如果线程已经完成更新操作
