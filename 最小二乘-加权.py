import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

angle_3 = [0.975, 1.002, 0.998, 1.06, 1.083, 1.0, 0.998, 0.998, 0.909, 0.904, 0.957, 1.002, 0.985, 1.002, 0.998]
detect_3 = [0.913, 0.957, 0.961, 1.064, 1.153, 1.017, 0.939, 1.031, 0.926, 1.014, 1.077, 1.028, 1.076, 0.985, 1.007]

# np.random.seed(10)
nsample = 100 # 点的个数
x = np.linspace(1, 100, nsample)

X = sm.add_constant(x)  # 添加常数列
X[:5]
beta = [5.0, 0.5, ]  # -0.01]
y_true = np.dot(X, beta)


# 添加残差
sig = 0.75
w = np.linspace(1, 10, nsample)
e = np.random.normal(size=nsample)
e = sig * w * e
y = y_true + e
X = X[:, [0, 1]]

plt.figure(figsize=(6, 4))
plt.scatter(x, y)

res_ols = sm.OLS(y, X).fit()
# print(res_ols.summary())


res_wls = sm.WLS(y, X, weights=1.0 / abs(res_ols.resid)).fit()
# print(res_wls.summary())

y_hat_ols = res_ols.params[0] + res_ols.params[1] * x

y_hat = res_wls.params[0] + res_wls.params[1] * x


plt.figure(figsize=(6, 4))
plt.scatter(x, y)

plt.plot(x, y_hat_ols, label='ols')
plt.plot(x, y_hat, label='wls')

plt.legend()
plt.show()
