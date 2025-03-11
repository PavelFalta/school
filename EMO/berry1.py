from random import choice, random
import string
 
target  = list("you should try coding and writing print('hello world') its not that hard and it will impress all you friends guarantee")
alphabet = string.ascii_lowercase + " " + "'" + "(" + ")"
 
pmut = 0.05
nchildren = 1000
 
def fitness(trial):
    return sum(t != h for t,h in zip(trial, target))
 
def mutate(parent):
    return [(choice(alphabet) if random() < pmut else ch) for ch in parent]
 
parent = [choice(alphabet) for _ in range(len(target))]
igen = 0
while parent != target:
    children = (mutate(parent) for _ in range(nchildren))
    parent = min(children, key=fitness)
    print(f"{igen} gen: {''.join(parent)}")
    igen += 1