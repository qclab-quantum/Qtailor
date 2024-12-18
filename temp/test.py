import concurrent.futures
import multiprocessing
import time

def FunctionA(index):
    # 模拟一个可能很耗时的操作
    print(f"FunctionA({index}) started")
    time.sleep(65)  # 假设这个函数需要65秒来执行
    print(f"FunctionA({index}) completed")
    return f"Result of FunctionA({index})"

def call_with_timeout(func, timeout, *args, **kwargs):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout)
            return result
        except concurrent.futures.TimeoutError:
            print(f"FunctionA({args[0]}) exceeded {timeout} seconds and was terminated")
            return None
        finally:
            # Ensure the executor is shut down properly
            executor.shutdown(wait=False)

def main():
    for i in range(5):  # 假设我们要调用5次
        result = call_with_timeout(FunctionA, 5, i)
        if result:
            print(result)
        else:
            print(f"Skipping FunctionA({i}) due to timeout")

if __name__ == "__main__":
    # To ensure multiprocessing works correctly on Windows
    multiprocessing.set_start_method('spawn')
    main()
