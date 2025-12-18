# Add to agent ops/fibonacci.py
def map_fibonacci(payload):
    """Calculate Fibonacci number at position n"""
    n = payload.get('n', 30)
    
    def fib(x):
        if x <= 1:
            return x
        return fib(x-1) + fib(x-2)
    
    import time
    start = time.time()
    result = fib(n)
    elapsed_ms = (time.time() - start) * 1000
    
    return {
        'n': n,
        'result': result,
        'compute_time_ms': elapsed_ms
    }
