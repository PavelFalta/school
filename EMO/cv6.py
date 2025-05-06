import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return -x**2 - 12

class Particle:
    def __init__(self, dim, prostor_od, prostor_do):
        
        self.pozice = np.random.uniform(prostor_od, prostor_do, dim)
        self.zrychleni = np.zeros(dim)
        self.nej_poz = self.pozice.copy()
        self.nej_fit = float('-inf')
    
    def evaluate(self):
        self.fitness = f(self.pozice)
        if self.fitness > self.nej_fit:  
            self.nej_fit = self.fitness
            self.nej_poz = self.pozice.copy()
        
        return self.nej_fit, self.nej_poz
    
    def update(self, nejlepsi_od_kamaradu, w, c1, c2):
        
        r1 = np.random.rand()
        r2 = np.random.rand()  

        c = c1 * r1 * (self.nej_poz - self.pozice)
        s = c2 * r2 * (nejlepsi_od_kamaradu - self.pozice)
        
        self.zrychleni = w * self.zrychleni + c + s
        self.pozice = self.pozice + self.zrychleni

def PSO():
    particles = [Particle(1, -20, 20) for i in range(100)]

    nejlepsi_od_kamaradu_pos = None
    nejlepsi_od_kamaradu_fit = float("-inf")

    for particle in particles:
        nej_fit, nej_pos = particle.evaluate()

        if nej_fit > nejlepsi_od_kamaradu_fit:
            nejlepsi_od_kamaradu_pos = nej_pos.copy()
            nejlepsi_od_kamaradu_fit = nej_fit.copy()

    for particle in particles:
        particle.update(nejlepsi_od_kamaradu_pos, w=0.5, c1=0.6, c2=0.4)
    
    return nejlepsi_od_kamaradu_pos, nejlepsi_od_kamaradu_fit


pos, fit = PSO()
print(f"nejlepsi pozice: {pos}, nejlepsi fit: {fit}")

t = np.linspace(-20, 20, 100)
plt.plot(t, f(t))
plt.scatter(pos, fit)
plt.show()