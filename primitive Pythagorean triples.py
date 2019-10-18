def GCD(n): 
    """Greatest common divisor"""
    a = n[0]
    for i in range(1, len(n)):
        b=n[i]
        while b:
            a,b = b, a % b
    if (a == 1):
        return True
    else:
        return False

end = 10000
f = open("Primitive Pythagorean_triples.txt", 'w')

for a in range(1, end + 1):
    if (a % 10 == 0):
        print(a)
    for b in range(a + 1, end + 1):
        c = (a ** 2 + b ** 2) ** 0.5
        if (a == 3 and b == 4 and c == 5):
            print("1st primitive Pythagorean triple: a = 3, b = 4, c = 5")
        n = a, b, c
        if (float(c).is_integer() == True and GCD(n) == True):
            triple = ("a = %d, b = %d, c = %d" % (a, b, c))  
            f.write(triple + "\n")
f.close()
