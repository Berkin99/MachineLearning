import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.svm import SVC

# 1. Veri Setini Oluştur
X, y = make_circles(n_samples=500, factor=0.5, noise=0.1)
# y: 0 ve 1 → SVM için -1 ve 1'e çevir
y = np.where(y == 0, -1, 1)

# 2. Doğrusal SVM (kernel trick yok)
clf = SVC(kernel='rbf', C=1.0, gamma='scale')

clf.fit(X, y)

# 3. Support Vector'leri al
support_vectors = clf.support_vectors_

# 4. Plot
plt.figure(figsize=(8, 6))

# Sınıf 1 ve -1 olanları ayır
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='b', label='Class +1')
plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='r', label='Class -1')

# Support vector'leri büyükçe dairelerle çiz
plt.scatter(support_vectors[:, 0], support_vectors[:, 1],
            s=150, facecolors='none', edgecolors='k', linewidths=1.5, label='Support Vectors')

plt.title('Linear SVM on Circular Data (No Kernel Trick)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
