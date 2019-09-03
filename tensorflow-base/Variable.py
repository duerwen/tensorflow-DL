import tensorflow as tf

# # 变量定义
# w = tf.Variable([[0.5,1.0]])
# x = tf.Variable([[2.0],[1.0]])
# # 定义操作
# y = tf.matmul(w,x)
#op_init = tf.global_variables_initializer()
# #变量必须初始化
# # tensorflow中的操作都要放到回话图中进行
# with tf.Session() as sess :
#     sess.run(op_init)
#     y_temp = sess.run(y)
#     print(y_temp)


# # 执行一个相加的操作
# state = tf.Variable(0)
# new_value = tf.add(state,tf.constant(1))
# update = tf.assign(state,new_value)
# op_init = tf.global_variables_initializer()
# with tf.Session() as sess :
#     sess.run(op_init)
#    # print(sess.run(state))
#     for i in range(3) :
#         print(sess.run(update))
#         #print(sess.run(state))


input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
out = tf.multiply(input1,input2)
with tf.Session() as sess :
    print(sess.run([out],feed_dict={input1:[7.0],input2:[2.0]}))


