'''
 =======================================================================
 ····Y88b···d88P················888b·····d888·d8b·······················
 ·····Y88b·d88P·················8888b···d8888·Y8P·······················
 ······Y88o88P··················88888b·d88888···························
 ·······Y888P··8888b···88888b···888Y88888P888·888·88888b·····d88b·······
 ········888······"88b·888·"88b·888·Y888P·888·888·888·"88b·d88P"88b·····
 ········888···d888888·888··888·888··Y8P··888·888·888··888·888··888·····
 ········888··888··888·888··888·888···"···888·888·888··888·Y88b·888·····
 ········888··"Y888888·888··888·888·······888·888·888··888··"Y88888·····
 ·······························································888·····
 ··························································Y8b·d88P·····
 ···························································"Y88P"······
 =======================================================================

 -----------------------------------------------------------------------
Author       : 焱铭
Date         : 2023-04-18 11:21:07 +0800
LastEditTime : 2023-09-15 15:43:03 +0800
Github       : https://github.com/YanMing-lxb/
FilePath     : \YM-MOP-NSGA2\NSGA-Ⅱ_pymoo-continuous-variable.py
Description  : 改代码用于处理变量全为连续变量的情况
 -----------------------------------------------------------------------
'''

import numpy as np
import NSGA2Post as post

from pymoo.core.problem import ElementwiseProblem

class MyProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=4,  # 表示x是一个长度为 n_var 的一维NumPy数组
                         n_obj=3,  # 目标值应该以长度为 n_obj 的NumPy数组列表的形式写到 out["F"]
                         xl=np.array([0, 0, 6, 0]),  # 表示变量下限，长度由 n_var 定义
                         xu=np.array([0.8, 0.8, 22, 16]))  # 表示变量上限，长度由 n_var 定义

    def _evaluate(self, x, out, *args, **kwargs):

        f1 = 2.265 + 1.075 * x[0] + 1.818 * x[1] + 0.01832 * x[2] + 0.0742 * \
            x[3] + 2.721 * x[0]*x[1] - 0.1042 * x[0]*x[3] - 0.1344 * x[1]*x[3]
        f2 = 2.11145 - 0.3793 * x[0] - 0.1643 * x[1] - 0.00368 * x[2] - 0.000824 * x[3] + 0.1425 * x[0]*x[0] + 0.0509 * x[1]*x[1] + 0.000109 * x[2]*x[2] - \
            0.000010 * x[3]*x[3] + 0.1211 * x[0]*x[1] + 0.001403 * x[0]*x[2] + 0.000857 * \
            x[0]*x[3] - 0.002922 * x[1]*x[2] + 0.001299 * \
            x[1]*x[3] + 0.000008 * x[2]*x[3]
        f3 = 0.6250 - 0.2196 * x[0] - 0.1265 * x[1] - 0.00571 * x[2] - 0.001057 * x[3] - 0.0512 * x[0]*x[0] - 0.0153 * x[1]*x[1] + 0.000159 * x[2]*x[2] + \
            0.000002 * x[3]*x[3] + 0.1390 * x[0]*x[1] + 0.002362 * x[0]*x[2] + 0.000806 * \
            x[0]*x[3] - 0.002073 * x[1]*x[2] + 0.001507 * \
            x[1]*x[3] + 0.000011 * x[2]*x[3]

        out["F"] = [f1, f2, f3]

        

problem = MyProblem()



# 初始化算法
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

algorithm = NSGA2(
    pop_size=200,  # 选择群体大小
    n_offsprings=100,  # 每代的后代个数
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True  # 启用重复检查，确保交配产生的后代与它们自己和现有种群的设计空间值不同。
)



# 定义终止标准
from pymoo.factory import get_termination

termination = get_termination("n_gen", 20)  # 以迭代次数为终止标准。



# 优化
from pymoo.optimize import minimize

res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=True)

X = res.X
F = res.F



# 数据处理
weight = [0.4,0.2,0.4] # 权重设置
NSGA2Post = post.NSGA2_post(X = X, F = F, history = res.history, HvDisplay='show') # 初始化函数，输入 HvDisplay='show' 可视化超立方体积法
NSGA2Post.Compromise_Programming(weight) # 均衡规划法(Compromise Programming)
NSGA2Post.Pseudo_Weights(weight) # 伪权向量法(Pseudo-Weights)
date_hv = NSGA2Post.Hypervolume() # 进行超立方体积计算
date_Topsis = NSGA2Post.Topsis() # 进行TOPSIS评分
NSGA2Post.export_excel(date_hv,date_Topsis) # 将分析完成的列表导出为excel表格
NSGA2Post.export_hist_excel(2) # 将每隔"gen"代数进行导出种群数据
NSGA2Post.ParetoDisplay() # 绘制出Pareto图




# from pymoo.util.reference_direction import UniformReferenceDirectionFactory
# from pymoo.visualization.pcp import PCP
# plot = PCP()
# plot.set_axis_style(color="grey", alpha=0.5)
# plot.add(F, color="grey", alpha=0.3)
# plot.add(F[40], linewidth=5, color="red")
# plot.add(F[12], linewidth=5, color="blue")
# plot.show()


