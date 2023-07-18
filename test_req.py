import generator
import time

if __name__ == '__main__':
    start = time.time()
    generator.get_data(100, 90)
    exec_time = time.time() - start
    print(exec_time)
