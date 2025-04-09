from cvxopt import matrix, solvers
import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(0, 10, 100)
y = x + np.random.normal(size=100) * 2

plt.scatter(x, y)
plt.show()

# svr implementation
def create_model_SVM_softm_primary(X, y, C=1):
    n,dim=X.shape #zjisti si rozmery
    # generovani matic a vektoru pro ucelovou funkci
    # generovani matice P - pro resic vstupuje jako matice prislusne kvadraticke formy
    P = np.zeros((dim+1+n,dim+1+n))

    for i in range(0, dim):
        P[i,i]=1 # prepis 1 na diagonalu pro hodnoty w
        
    # generovani vektoru q - vektor s koeficienty pro linearni cast ucelove funkce
    q = C*np.ones(dim+1+n) # linearni clen v ucelove funkci soucet ksi
    q[:dim+1] = 0 #nastav na nulu pozice w a b ve vektoru q

    # generovani omezujicich podminek
    h = np.concatenate((-1*np.ones((n,1)), np.zeros((n,1))))

    G = np.zeros((2*n, n+dim+1))
    for i,(x,y) in enumerate(zip(X,y)):
        G[i,:dim] = -y*x # sloupce odpovidajici w
        G[i,dim] = -y  # sloupec odpovidajici b
        G[i,dim+1+i] = -1
        G[i+n,dim+1+i] = -1
    print(f"P:{P.shape}")
    print(f"q:{q.shape}")
    print(f"G:{G.shape}")
    print(f"h:{h.shape}")
    return matrix(P), matrix(q), matrix(G), matrix(h)
# vygeneruj si matice modelu
P, q, G, h = create_model_SVM_softm_primary(X,y, C = 10)
# volej rešič
sol = solvers.qp(P, q, G, h)
w = sol['x']
#vizualizace výsledků
left = np.min(X, axis = 0)
right = np.max(X, axis = 0)
xgr = np.linspace(left[0], right[0],100)
ygr = -(w[0]*xgr + w[2])/w[1]
ygr_p1 = -(w[0]*xgr + w[2]+1)/w[1]
ygr_m1 = -(w[0]*xgr + w[2]-1)/w[1]
plt.scatter(X[:,0],X[:,1], c = y)
plt.plot(xgr,ygr)
plt.plot(xgr,ygr_p1, label = "+1")
plt.plot(xgr,ygr_m1, label = "-1")
plt.grid()
plt.title("Body různých tříd v prostoru příznaků - soft margin")
plt.show()