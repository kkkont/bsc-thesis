import time

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def warm_up(duration_sec=300):
    print(f"Starting warm-up for {duration_sec} seconds...")
    end_time = time.time() + duration_sec
    i = 30  
    while time.time() < end_time:
        fibonacci(i)
    print("Warm-up complete.")

warm_up()
