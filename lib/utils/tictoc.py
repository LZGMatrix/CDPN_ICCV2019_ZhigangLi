import time

def tic():
    global start_time
    start_time = time.time()
    return start_time

def toc():
    if 'start_time' in globals():
        end_time = time.time()
        return end_time - start_time
    else:
        return None