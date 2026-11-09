import numpy as np
import matplotlib.pyplot as plt

# ====================== 网络超参，全部固定，只换初始化 ======================
in_dim = 128
h_dim = 256
out_dim = 10
n_sample = 200

# 随机输入，固定种子保证可复现
np.random.seed(42)
X = np.random.randn(n_sample, in_dim)

def relu(z):
    return np.maximum(0, z)

# 四种初始化函数
def init_zero(inp, hid):
    return np.zeros((inp, hid)), np.zeros(hid)

def init_small_random(inp, hid, sigma=0.01):
    w = np.random.normal(0, sigma, size=(inp, hid))
    b = np.zeros(hid)
    return w,b

def init_xavier(inp, hid):
    scale = np.sqrt(2.0/(inp+hid))
    w = np.random.normal(0, scale, size=(inp, hid))
    b = np.zeros(hid)
    return w,b

def init_he(inp, hid):
    scale = np.sqrt(2.0 / inp)
    w = np.random.normal(0, scale, size=(inp, hid))
    b = np.zeros(hid)
    return w,b

init_methods = {
    "zero": init_zero,
    "small_random(σ=0.01)": init_small_random,
    "Xavier": init_xavier,
    "He": init_he
}

records = []

for name, init_func in init_methods.items():
    np.random.seed(42)
    W1, b1 = init_func(in_dim, h_dim)
    W2, b2 = init_he(h_dim, out_dim) #第二层固定用He，只改变W1初始化

    #前向传播
    h_pre = X @ W1 + b1
    h_act = relu(h_pre)

    #模拟简单梯度（简单反向，统计梯度分布）
    dL_dh = np.random.randn(n_sample, h_dim) * 0.1
    dL_dW1 = X.T @ dL_dh / n_sample

    #统计激活、梯度的均值、方差
    h_mean = np.mean(h_act)
    h_var = np.var(h_act)
    grad_mean = np.mean(dL_dW1)
    grad_var = np.var(dL_dW1)

    rec = {
        "init_type": name,
        "activation_mean": h_mean,
        "activation_var": h_var,
        "grad_mean": grad_mean,
        "grad_var": grad_var
    }
    records.append(rec)
    print(f"==== {name} ====")
    print(f"激活值 mean:{h_mean:.4f}, var:{h_var:.4f}")
    print(f"W1梯度 mean:{grad_mean:.4f}, var:{grad_var:.4f}\n")

# 把结果打印表格，周二基于这些统计量画对比图
import pandas as pd
df = pd.DataFrame(records)
print(df)

# ===================== 绘图 ====================
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,5))

names = df["init_type"].values
act_var = df["activation_var"].values
grad_var = df["grad_var"].values

ax1.bar(names, act_var, color=["#222222","#4477aa","#22aa77","#dd7733"])
ax1.set_title("不同初始化：第一层ReLU激活方差对比")
ax1.set_ylabel("激活方差")
ax1.tick_params(axis='x', rotation=15)

ax2.bar(names, grad_var, color=["#222222","#4477aa","#22aa77","#dd7733"])
ax2.set_title("不同初始化：W1梯度方差对比")
ax2.set_ylabel("梯度方差")
ax2.tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig("init_compare.png", dpi=200)
plt.show()
