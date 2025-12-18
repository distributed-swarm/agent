# Add to agent ops/prime_factor.py
def map_prime_factor(payload):
    """Factorize a large number into primes"""
    n = int(payload.get('number', 0))
    
    import time
    start = time.time()
    
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    
    elapsed_ms = (time.time() - start) * 1000
    
    return {
        'original': payload.get('number'),
        'factors': factors,
        'compute_time_ms': elapsed_ms
    }
