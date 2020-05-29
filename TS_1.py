import random
import numpy as np

# 路径长度函数，tour表示路径
def getTourLength(sol, distance):
    distance_ = distance.copy()
    sol_ = sol.copy()
    n = len(sol_)
    length = 0
    sol_.append(sol_[0])
    for k in range(n):
        i = sol_[k]
        j = sol_[k + 1]
        length += distance_[i, j]
    length =round(length, 10)
    return length

# 计算距离矩阵
def getDistance(x, y):
    n = len(x)
    distance = np.zeros((n, n))
    for i in range(n - 1):
        for j in range(i, n):
            distance[i, j] = np.sqrt(pow((x[i] - x[j]), 2) + pow((y[i] - y[j]), 2))
    distance += distance.T - np.diag(distance.diagonal())
    return distance

def getActionlist(n):
    actionList = []
    c = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            c += 1
            actionList.append([1, i, j])
    for i in range(n - 1):
        for j in range(i + 1, n):
            if abs(i - j) > 2:
                c += 1
                actionList.append([2, i, j])
    return actionList

# 执行动作例表，其中p为当前解，a为执行的动作，a的格式为[x1,x2,x3],x1表示选择的领域操作类型,x1= 1时为交换，x1=2时为逆转，x2、x3表示位置索引。
def doAction(p, a):
    if a[0] == 1:  # 执行交换操作
        q = p.copy()
        q[a[1]] = p[a[2]]
        q[a[2]] = p[a[1]]
    if a[0] == 2:  # 执行逆转操作
        q = p.copy()
        if a[1] < a[2]:
            reversion = p[a[1]:a[2] + 1]
            reversion.reverse()
            q[a[1]:a[2] + 1] = reversion
        else:
            reversion = p[a[2]:a[1] + 1]
            reversion.reverse()
            q[a[2]:a[1] + 1] = reversion
    return q

# 请在此处添加代码，实现目标函数功能
# ********** Begin **********#

def getBestSol(ActionList, sol, TC, TL, distance, MaxIt):
    action_len = len(ActionList)
    TC = list(TC)
    for it in range(MaxIt):
        dist_list = []
        cur_len = getTourLength(sol,distance)
        for i in range(action_len):
            cur_sol = sol.copy()
            next_sol = doAction(cur_sol,ActionList[i])
            next_len = getTourLength(next_sol,distance)
            dist_list.append(next_len)
        sort_index = np.argsort(dist_list)
        del TC[0]
        for i in range(TL):
            if all(ActionList[sort_index[i]]) not in TC:
                TC.append(ActionList[sort_index[i]])
                break
        sol = doAction(sol, ActionList[sort_index[i]])
        if dist_list[sort_index[i]] <= cur_len:
            best_sol = sol.copy()

    return best_sol

# *********** End ***********#

x = [82, 91, 12, 92, 63, 9, 28, 55, 96, 97, 15, 98, 96, 49, 80, 12, 14, 41, 24, 5, 73, 56, 5, 94, 11]
y = [14, 42, 92, 80, 96, 66, 3, 85, 94, 68, 76, 75, 39, 66, 17, 78, 54, 74, 8, 25, 12, 1, 15, 52, 28]

ActionList = getActionlist(len(x))
sol = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]  # 当前解
nAction = len(ActionList)  # 操作总数
TL = round(0.02 * nAction)  # 禁忌长度
MaxIt = 1000 # 最大迭代次数
TC = np.zeros(TL)  # 初始化禁忌表
distance = getDistance(x, y)  # 得到距离矩阵
best_sol= getBestSol(ActionList, sol, TC, TL, distance, MaxIt)  # 目标函数
print(best_sol)
# 输出结果
