# Add to agent ops/subset_sum.py
def map_subset_sum(payload):
    """Find subset of numbers that sum to target"""
    numbers = payload.get('numbers', [])
    target = payload.get('target', 0)
    
    import time
    start = time.time()
    
    # Dynamic programming solution
    n = len(numbers)
    dp = [[False] * (target + 1) for _ in range(n + 1)]
    
    for i in range(n + 1):
        dp[i][0] = True
    
    for i in range(1, n + 1):
        for j in range(target + 1):
            if j < numbers[i-1]:
                dp[i][j] = dp[i-1][j]
            else:
                dp[i][j] = dp[i-1][j] or dp[i-1][j - numbers[i-1]]
    
    # Backtrack to find subset
    subset = []
    if dp[n][target]:
        i, j = n, target
        while i > 0 and j > 0:
            if dp[i-1][j]:
                i -= 1
            else:
                subset.append(numbers[i-1])
                j -= numbers[i-1]
                i -= 1
    
    elapsed_ms = (time.time() - start) * 1000
    
    return {
        'found': dp[n][target],
        'subset': subset,
        'sum': sum(subset),
        'target': target,
        'compute_time_ms': elapsed_ms
    }
