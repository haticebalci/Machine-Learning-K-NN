'''Python içine gerekli kütüphaneler import edilir.Diğer kütüphaneler ilgili işlem yapılmadan önce aşağıda import edilecektir. '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''Kullanacağımız veri seti Iris veri setidir.Python'ın pandas kütüphanesinin read_excel methodu ile veri setini import ediyoruz.'''

data=pd.read_excel('Iris.xls')
print(data)

'''Iris veri seti toplamda 5 kolondan oluşmaktadır.Kolonlardan biri bağımlı değişken diğerleri ise bağımsız değişkenlerdir.Bağımsız değişken kolonlarda verilen 
ölçüm özelliklerine species kolonu için sınıflandırma yapacağız.Öncesinde bağımsız değişkenlerdeki nitelikler için bir x matrisi,bağımlı değişken için ise bir y vektörü 
oluşturacağız.'''

X=data.iloc[:,0:-1]
Y=data.iloc[:,4:] 

'''Bağımlı ve bağımsız değişkenlerimizi belirledikten sonra Iris veri seti 4 bölüme ayrılır.Bu bölümlerden %67'lik kısım olan X_train ve Y_train eğitim için kullanılırken
%33'lük kısım olan  X_test ve Y_test ise makineye tahmin ettirilmeye çalışılacaktır.'''

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.33,random_state=0)

'''Modele geçmeden önce verilerde Standard Scaler denilen ön işleme yapılır.Standart ölçeklendirme işlemi, verilerin her bir özelliğini ortalama değerinden çıkarmak ve standart sapmaya 
bölmek suretiyle gerçekleştirilir. Bu, verilerinizi merkezileştirir ve standart sapmaya göre ölçeklendirir, böylece her bir özellik ortalama değeri sıfır ve standart sapması bir olan 
bir dağılım şekline sahip olur.Bu ölçeklendirme yöntemi, özellikle makine öğrenimi algoritmaları gibi modellerin performansını arttırmak için kullanılan ön işleme adımlarından biridir. 
Verilerin ölçeklendirilmesi,modelin daha doğru ve güvenilir sonuçlar üretmesine yardımcı olur ve aynı zamanda modelin eğitim sürecini de hızlandırır.'''

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

'''Makine öğrenmesi sınıflandırma algoritmalarından bir tanesi de K-NN'dir.K-NN algoritmasının temel mantığı kendisine en yakın olan komşularını 
kullanarak yeni örnekleri sınıflandırmasıdır. Bir uzaklık ölçütü kullanarak kendisine en yakın komşularını belirler.K-NN algoritmasında 2 temel değer üzerinden tahminleme yapılır.
1)k(n_neighbors)=En yakın kaç komşu üzerinden hesaplama yapılacağı k değeri ile belirlenir.Sonuca direkt olarak etki eden k değeri optimum olarak belirlenmelidir.Ne çok yüksek bir 
k değeri ne de çok düşük bir k değeri olmamasına dikkat edilerek bu değer belirlenmelidir.K değeri belirlenmediyse default olarak 5 olarak hesap edilir.

2)Distance=Tahmin edilecek olan noktanın diğer noktalara olan uzaklığı hesap edilir.Distance kendi içinde birçok hesaplama metdou olup belirlenmediği takdirde default olarak 'minkowski'
olarak belirlenecektir.'''


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3,metric='minkowski',p=2)
knn.fit(X_train, Y_train.values.ravel())
tahmin=knn.predict(X_test)

'''Confusion matrix,sınıflandırma problemlerinde kullanılan bir performans ölçümüdür. Karışıklık matrisi, gerçek sınıfı ve tahmin edilen sınıfı içeren bir tablodur. 
Bu tablo, dört farklı değere sahip olabilir: true positive (TP), false positive (FP), true negative (TN) ve false negative (FN).TP, modelin doğru bir şekilde bir sınıfı
 belirlediği durumlarda oluşurken, FP modelin yanlış bir şekilde bir sınıfı belirlediği durumlarda oluşur.TN, modelin bir sınıfı doğru bir şekilde olmadığını belirlediği 
 durumlarda, FN ise modelin bir sınıfı yanlış bir şekilde olmadığını belirlediği durumlarda oluşur.Karmaşıklık matrisi, bu dört sonucu bir matris içinde gösterir.''' 


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, tahmin)
print(cm)
#Confusion matrix:
#[[16  0  0]
# [ 0 18  1]
# [ 0  1 14]]

''' Accuracy, doğru sınıflandırılmış örneklerin toplam sayısının tüm örneklerin toplam sayısına oranıdır.Accuracy = (TP + TN) / (TP + FP + TN + FN)
Accuracy, sınıflandırma modelinin tüm sınıfları doğru bir şekilde tahmin etme becerisini ölçer. Ancak, dengesiz sınıf dağılımları gibi durumlarda yanıltıcı olabilir 
ve diğer performans metrikleri ile birlikte kullanılması önerilir.'''

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, tahmin)
print(accuracy)
#Başarı oranı:0.96 olup 50 veriden 48 tanesi doğru tahmin edilmiştir.

'''Accuracy değeri gibi f1 skor da sınıflandırma algoritmalarının başarısını ölçmek için kullanılan bir metriktir.Özellikle dengeli olmayan veri setleri için sınıflandırma problemlerinde
kullanılı bir metriktir.F1 skor 0 ve 1 arasında bir değer alıp 1 en iyi 0 ise en kötü performansa sahip olunduğu anlamına gelmektedir.'''

from sklearn.metrics import f1_score
f1=f1_score(Y_test, tahmin,average='macro')
print(f1)
#f1 score:0.9602339181286549

