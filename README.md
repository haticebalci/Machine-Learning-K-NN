# Machine-Learning-K-NN
# K-NN(K-Nearest Neighbors)
Bu, Iris veri kümesi üzerinde K-En Yakın Komşular (K-NN) algoritmasını uygulamak için bir Python kodudur.
Kodun ne yaptığına dair kısa bir özet şöyledir:

- Gerekli kütüphaneler, pandas, numpy ve matplotlib dahil olmak üzere içe aktarılır.
- Iris veri kümesi, pandas kütüphanesinin read_excel yöntemi kullanılarak içe aktarılır.
- Bağımsız ve bağımlı değişkenler sırasıyla X ve Y olarak tanımlanır.
- Veri kümesi, sklearn'in train_test_split yöntemi kullanılarak eğitim ve test kümelerine ayrılır.
- Veriler, sklearn.preprocessing'in StandardScaler yöntemi kullanılarak standartlaştırılır.
- K-NN algoritması, sklearn.neighbors'taki KNeighborsClassifier yöntemi kullanılarak uygulanır.
- Karışıklık matrisi, sklearn.metrics'teki confusion_matrix yöntemi kullanılarak hesaplanır.
- Genel olarak, kod verileri önişler, K-NN algoritmasını eğitir ve ardından test kümesinde algoritmayı test ederek performans metriği olarak karışıklık matrisi üretir.
