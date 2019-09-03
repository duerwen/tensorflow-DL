import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 生成数据(生成[0,1）的随机样本)
x_data = np.random.rand(100)
print(x_data)
# 生成符合正态分布的噪声数据(参数均值，标准差，数据形状)
nosie = np.random.normal(0,0.01,x_data.shape)
y_data = 0.2*x_data+0.1+nosie
plt.scatter(x_data,y_data)
plt.show()

# 定义模型
d = tf.Variable(np.random.rand(1))
k = tf.Variable(np.random.rand(1))
y = k*x_data+d
# 定义均方误差
loss = tf.losses.mean_squared_error(y,y_data)
# 定义最小化均方误差的方式(梯度下降法)
optimizer = tf.train.GradientDescentOptimizer(0.3)
# 最小化误差
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
# 定义一个操作来执行
with tf.Session() as session :
    # 变量一定要初始化
    session.run(init)
    for i in range(201) :
        session.run(train)
    y_pred  = session.run(y)
    plt.scatter(x_data,y_data)
    plt.plot(x_data,y_pred,"r")
    plt.show()
