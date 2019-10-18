low = int(input("low: "))
high = int(input("high: "))

def binary_search(low, high):
    while (low < high):
        mid = (high + low) // 2
        print(mid)
        diff = str(input("diff: "))
        if diff == '<':
            high = mid - 1
        elif diff == '>':
            low = mid + 1
        else:
            return None
    return high

print(binary_search(low, high))
