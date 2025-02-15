from time import perf_counter
def premenit_jmena(arr):
    start = perf_counter()
    counter = 0
    res = []
    for jmeno in arr:
        b = []
        for i, letter in enumerate(jmeno):

            if counter % 3 == 0:
                b.append(jmeno[i].upper())
            else:
                b.append(jmeno[i])
            
            counter += 1
        res.append("".join(b))



    return perf_counter()-start

def premenit_jmena2(arr):
    start = perf_counter()
    
    velky_list = [ch for name in arr for ch in name]
    
    for i in range(0, len(velky_list), 3):
        velky_list[i] = velky_list[i].upper()
    
    res = ["".join(velky_list[i:i+len(name)]) for name, i in zip(arr, range(0, len(velky_list), len(arr[0])))]

    return perf_counter() - start


long_name_list = ["tomas", "jana", "pavel"]*1000000

print(premenit_jmena(long_name_list))
print(premenit_jmena2(long_name_list))