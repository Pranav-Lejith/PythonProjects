import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} executed in {end - start} seconds")
        return result
    return wrapper

@timer
def example_function(n):
    sum = 0
    for i in range(n):
        sum += i
    return sum

print(example_function(1000000))
