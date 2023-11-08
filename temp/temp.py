import time

start_time = time.time()

time.sleep(1)
# 获取执行时间
end_time = time.time()
runtime = round(end_time - start_time,3)
print("Function runtime:", runtime, "seconds")