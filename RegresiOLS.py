import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import t,f,kstest,norm,probplot


class ModelRegresi:

    def __init__(self, x, y):
        if isinstance(x, (pd.Series, pd.DataFrame, list)) and isinstance(y, (pd.Series, pd.DataFrame, list)):
            self.X = np.column_stack((np.ones(len(x), dtype=int), np.array(x)))
            self.y = y
        else:
            self.X = np.column_stack((np.ones(len(x), dtype=int), np.array(x)))
            self.y = y

    def ModelOLS(self):
        XTX = np.dot(self.X.T, self.X)
        XTX_invers = np.linalg.pinv(XTX)
        XTY = np.dot(self.X.T, self.y)
        beta = np.dot(XTX_invers, XTY)
        return beta
    
    def YpredOLS (self):
        y_pred = self.X @ self.ModelOLS()
        return y_pred
    
    def ResidualOLS (self):
        Residual = self.y - self.YpredOLS()
        return Residual
    
    def Sd (self):
        X = np.delete(self.X, 0, axis = 1)
        sd = np.sqrt(np.sum(self.ResidualOLS()**2) / (len(self.y) - X.shape[1]))
        return sd
    
    def laverage (self):
        X = np.delete(self.X, 0, axis = 1)
        invers = np.linalg.pinv(X.T @ X)
        nilai_laverage_hi = np.diagonal(X @ invers @ X.T)
        return nilai_laverage_hi
    
    def DFFITS (self):
        hasil1 = (self.YpredOLS() - self.y) 
        hasil2 = (self.Sd() * np.sqrt(1 - self.laverage())).reshape(-1,1)
        return hasil1/hasil2.flatten()
    
    def BoxPLotPencilan(self):

        nama_kolom = []
        for i in range(self.X.shape[1]):
            if i == 0:
                label = f"variabel intecept"
                nama_kolom.append(label)
            else:
                label = f"variabel {i}"
                nama_kolom.append(label)

        plt.figure(figsize=(10, 5))  # Ukuran gambar
        plt.boxplot(self.X, labels=nama_kolom)
        plt.title('Box Plot Masing-masing variabel')
        plt.ylabel('Nilai')
        plt.show()
        
    def Kriteria_lav(self):
        X = np.delete(self.X, 0, axis = 1)
        kriteria_laverage = 2 * (X.shape[1]) / X.shape[0]
        return kriteria_laverage
    
    def Kriteria (self):
        X = np.delete(self.X, 0, axis = 1)
        kriteria = 2 * np.sqrt(X.shape[1] / X.shape[0])
        return kriteria
    
    def TblPencilan (self):
        keterangan_lav = []
        keterangan_DFFITS = []
        X = np.delete(self.X, 0, axis = 1)
        kriteria_lav = self.Kriteria_lav()
        laverage = self.laverage()
        kriteria = self.Kriteria()
        dffits = self.DFFITS()
        columns = [f"X{i+1}" for i in range(len(X[1]))]

        tabel_pencilan = pd.DataFrame(X, columns = columns)
        tabel_pencilan['laverage'] = self.laverage()
        tabel_pencilan['kriteria_lav'] = self.Kriteria_lav()
        for k in laverage:
            if k < kriteria_lav:
                keterangan = "bukan pencilan"
                keterangan_lav.append(keterangan)
            else:
                keterangan = "pencilan"
                keterangan_lav.append(keterangan)
        tabel_pencilan['ket_laverage'] = keterangan_lav
        tabel_pencilan['DFFITS'] = self.DFFITS()
        tabel_pencilan['kriteria'] = self.Kriteria()
        for x in dffits:
            if abs(x) < kriteria:
                ket ="bukan pencilan"
                keterangan_DFFITS.append(ket)
            else:
                ket = "pencilan"
                keterangan_DFFITS.append(ket)
        tabel_pencilan['ket_DFFITS'] = keterangan_DFFITS
        return tabel_pencilan
    
    def Rsquare(self):
        a = np.sum((self.y - np.mean(self.y)) ** 2)
        b = np.sum((self.y - self.YpredOLS()) ** 2)
        return 1- (b/a)
    
    def RamalOLS(self, data):
        hasil = np.insert(data, 0, 1)
        y_pred = hasil @ self.ModelOLS()
        return y_pred
    
    #thitung
    
    def ParsialTest(self):
        X = np.delete(self.X, 0, axis=1)
        keterangan = []
        def tabel_t():
            sig = 0.05
            df = (2*X.shape[0]) - 3
            t_tabel = t.ppf(1-sig,df)
            return t_tabel
        beta_hat = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y
        residual = self.y - (self.X @ beta_hat)
        n = self.X.shape[0]
        k = self.X.shape[1] - 1
        sigma_squared = np.sum(residual ** 2) / (n - k - 1)
        XTX_inverse = np.linalg.inv(self.X.T @ self.X)
        SE_beta = np.sqrt(sigma_squared * np.diag(XTX_inverse))
        p_values = 2 * (1 - t.cdf(np.abs(beta_hat/SE_beta), n - k - 1))
        thitung = beta_hat/SE_beta
        p_values = 2 * (1 - t.cdf(np.abs(beta_hat/SE_beta), n - k - 1))
        df = pd.DataFrame(index = [f"x{i+1}" for i in range(self.X.shape[1])])
        df['coefisient'] = thitung
        t_tabel = tabel_t()
        df['ttabel'] = t_tabel
        for i, x in enumerate(thitung):
            if i == 0:
                ket = 'signifikan'
                keterangan.append(ket)
            elif x > t_tabel:
                ket = 'signifikan'
                keterangan.append(ket)
            else:
                ket = 'tidak signifikan'
                keterangan.append(ket)
        df['keterangan'] = keterangan
        df['p vlaue'] = p_values
        return df
    
    
    def SimultanTest(self):
        beta_hat = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y
        y_pred = self.X @ beta_hat
        residual = self.y - y_pred
        SSR = np.sum((y_pred - np.mean(self.y)) ** 2)
        SSE = np.sum(residual ** 2)
        k = self.X.shape[1] - 1
        n = self.X.shape[0]
        F = (SSR / k) / (SSE / (n - k - 1))
        df1 = k
        df2 = n - k - 1
        p_value = 1 - f.cdf(F, df1, df2)
        df = pd.DataFrame({'f': [F], 'p_value' : [p_value]})
        df['keterangan'] = 'signifikan' if p_value < 0.05 else 'tidak signifikan'

        return df
    
    def NormTest (self, method):
        mean = np.mean(self.ResidualOLS())
        std = np.std(self.ResidualOLS())
        if method == 'SW':
            statistic_w, p_value = stats.shapiro(self.ResidualOLS())
            hasil = pd.DataFrame({'W': [statistic_w], 'p_value': [p_value]})
            return hasil
        
        elif method == 'KS':
            statistic_ks, pvalue = kstest(self.ResidualOLS(), 'norm', args = (mean, std))
            hasil = pd.DataFrame({'statistic_ks': [statistic_ks], 'pvalue' : [pvalue]})
            return hasil
        
        elif method == 'PlotNormalHist':
            plt.hist(self.ResidualOLS(), bins=15, density=True, alpha=0.6, color='g', label='Residual')
            x = np.linspace(min(self.ResidualOLS()), max(self.ResidualOLS()), 100)
            plt.plot(x, norm.pdf(x, mean, std), 'r', label='Distribusi Normal')
            plt.xlabel('Nilai')
            plt.ylabel('Frekuensi Normalized')
            plt.legend()
            plt.title('Histogram dan Distribusi Normal')
            plt.show()

        elif method == 'QqPlotNormal':
            probplot(self.ResidualOLS(), plot=plt)
            plt.plot([-2.4,2.5],[-2.5,2.5], 'r--')
            plt.title("QQ plot - Normal Residual")
            plt.show()

        else:
            return 'method tidak di temukan :(. Silahkan masukan kembali(SW,KS,PlotNormalHist dan QqPlotNormal)'
        
    def heteroskedastisitas (self):
        beta = np.linalg.pinv(self.X.T @ self.X) @ self.X.T @ self.y
        y_pred = self.X @ beta
        residu = self.y - y_pred
        X_kuadrat = self.X**2
        residu2 = residu**2
        beta2 = np.linalg.pinv(X_kuadrat.T @ X_kuadrat) @ X_kuadrat.T @ residu2
        ypred2 = X_kuadrat @ beta2
        a = np.sum((residu2 - np.mean(residu2)) ** 2)
        b = np.sum((residu2 - ypred2) ** 2)
        r2=1-(b/a)
        n = self.X.shape[0]
        BP = n * r2
        df_chi_square = self.X.shape[1]-1
        p_value = 1 - stats.chi2.cdf(BP, df_chi_square)
        # Cetak hasil
        print("Uji Breusch-Pagan:")
        print("Nilai R-squared tambahan:", r2)
        print("Nilai uji Breusch-Pagan:", BP)
        print("P-value:", p_value)
        
    def Aotokorelasi (self):
        def durbin_watson(residual):
            diff_residual = np.diff(residual) 
            sum_squared_diff = np.sum(diff_residual ** 2)  
            sum_squared_residual = np.sum(residual ** 2)  
            DW = sum_squared_diff / sum_squared_residual  
            return DW

        dw_value = durbin_watson(self.ResidualOLS())
        return f"nilai Duwbin Watson : {dw_value}"
        
    def MultikoTest(self, method):
        
        X = np.delete(self.X, 0, axis=1)
        if method == 'heatmap':
            def hitung_korelasi(x, y):
                n = len(x)
                mean_x = np.mean(x)
                mean_y = np.mean(y)
                atas = sum((x - mean_x) * (y - mean_y))
                bawah = np.sqrt(sum((x - mean_x) ** 2) * sum((y - mean_y) ** 2))
                cor = atas / bawah
                return cor

            variabel = X.shape[1]
            kor = np.zeros((variabel, variabel))

            for i in range(variabel):
                for j in range(i, variabel):
                    x = X[:, i]
                    y = X[:, j]
                    korelasi = hitung_korelasi(x, y)
                    kor[i, j] = korelasi
                    kor[j, i] = korelasi
            print(kor)

            x_labels = [f"x{i+1}" for i in range(kor.shape[1])]
            y_labels = [f"x{i+1}" for i in range(kor.shape[0])]
            sns.heatmap(kor, annot=True, cmap='YlGnBu', linewidths=0.5, square=True, fmt='.2f', xticklabels=x_labels, yticklabels=y_labels)
            plt.show()
        
        elif method == 'VIF':
            X = np.delete(self.X, 0, axis=1)
            variabel = X.shape[1]
            VIF = np.zeros(variabel)
            for i in range(variabel):
                n = np.delete(X, i, axis=1)  
                y = X[:, i]  
                beta = np.linalg.inv(n.T @ n) @ n.T @ y
                y_hat = n @ beta
                SSR = sum((y_hat - np.mean(y)) ** 2)
                SSE = sum((y - y_hat) ** 2)
                SST = SSR + SSE
                R_squared = SSR / SST
                VIF[i] = 1 / (1 - R_squared)
                df = pd.DataFrame(VIF.reshape(1,-1), columns= [f"x{i+1}" for i in range(X.shape[1])])
            return df
        else:
            return 'method yang dimasukan tidak ditemukan :(, silahkan masukan method yang benar(heatmap, VIF)'
    
    def ScaterPlotXY(self):
        X = np.delete(self.X, 0, axis=1)
        num = X.shape[1]

        def warna_acak():
            r = random.randint(0, 255) / 255.0
            g = random.randint(0, 255) / 255.0
            b = random.randint(0, 255) / 255.0
            return (r, g, b)

        for i in range(num):
            color = warna_acak()
            slope, intercept = np.polyfit(X[:,i], self.y, 1)
            x_regresi = np.linspace(np.min(X[:,i]), np.max(X[:,i]), 100)
            y_regresi = slope * x_regresi + intercept
            plt.figure()
            plt.plot(x_regresi, y_regresi, color='red', label='Regresi')
            plt.scatter(X[:, i], self.y, s=30, marker='o', c=color, label=f"variabel x{i+1}")
            plt.legend()
            plt.show()

    def PlotModel (self):
        plt.plot(self.y, color = 'blue', label = 'y aktual')
        plt.plot(self.YpredOLS(), color = 'red', label = 'y pred')
        plt.legend()
        plt.show()
        
