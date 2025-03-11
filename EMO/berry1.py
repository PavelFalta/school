from random import choice, random
import string
 
target  = list("print('hello world')")
alphabet = string.ascii_lowercase + " " + "'" + "(" + ")"
 
pmut = 0.05
nchildren = 100
 
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