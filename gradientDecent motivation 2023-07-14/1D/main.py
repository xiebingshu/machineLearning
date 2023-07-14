import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist
import numpy as np
import mpl_toolkits.mplot3d
x, y = np.mgrid[-2:2:20j, -2:2:20j]
z = (x ** 2 + y ** 2)
ax = plt.subplot(111, projection='3d')
ax.set_title('f(x,y) = x^2 + y^2')
ax.plot_surface(x, y, z, rstride=9, cstride=1, cmap=plt.cm.Blues_r)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()


def grad_2(p):
    derivx = 2 * p[0]
    derivy = 2 * p[1]
    return np.array([derivx, derivy])
#  p是一组参数的值


def grad_descent(grad, p_current, learing_rate, precision, iters_max):
    for i in range(iters_max):
        print('第', i, '次迭代的p值为：', p_current)
        grad_current = grad(p_current)
        if np.linalg.norm(grad_current, ord=2)<precision:
            break
        else:
            p_current = p_current - grad_current * learing_rate
    print('最小值处p为', p_current)
    return p_current


if __name__ == '__main__':
    grad_descent(grad_2, p_current=np.array([1,-1]), learing_rate=0.1, precision=0.000001, iters_max=10000)