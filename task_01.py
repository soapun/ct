def count_unique_symbols(it):
    alphabet = [0 for i in range(52)]
    res = 0
    
    for i in it:
        res += not alphabet[ord(i) - ord('a')]
        alphabet[ord(i) - ord('a')] = 1
        
    return res        

if __name__ == "__main__":
    print(count_unique_symbols(input()))