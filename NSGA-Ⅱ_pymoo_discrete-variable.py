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
Date         : 2023-04-18 18:45:31 +0800
LastEditTime : 2023-09-15 15:44:53 +0800
Github       : https://github.com/YanMing-lxb/
FilePath     : \YM-MOP-NSGA2\NSGA-Ⅱ_pymoo_discrete-variable.py
Description  : 该代码用于处理变量全为离散变量或混合变量的情况
 -----------------------------------------------------------------------
'''

import NSGA2Post as post
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Binary, Choice, Integer, Real


# 定义优化函数
class My_problem(ElementwiseProblem):

    def __init__(self, **kwargs):
        vars = {
            "H_rib": Integer(bounds=(16, 80)), # 整数类型 # 缩放之前的数值
            "H_pf": Integer(bounds=(16, 80)),
            "N_pf": Integer(bounds=(6, 22)), # 整数类型
            "N_ac": Integer(bounds=(0,8)) # 奇数 
            }                                                                                                                                                                                                                                                                                                                                      
        super().__init__(vars=vars, # 定义变量字典
                         n_obj=3, # 目标值应该以长度为 n_obj 的NumPy数组列表的形式写到 out["F"]
                         n_ieq_constr=0, 
                         **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # 公式进行缩放，缩放后的因变量应为真实变量
        H_rib,H_pf,N_pf,N_ac = X["H_rib"]/100, X["H_pf"]/100, X["N_pf"], X["N_ac"]*2
        # 目标1
        f1 = 1.9919 - 0.3049 * H_rib - 0.0715 * H_pf - 0.00407 * N_pf + 0.009650 * N_ac + 0.1487 * H_rib*H_rib + 0.000124 * N_pf*N_pf + 0.1059 * H_rib*H_pf + 0.001303 * H_rib*N_pf - 0.005765 * H_rib*N_ac- 0.002577 * H_pf*N_pf - 0.002381 * H_pf*N_ac

        # 目标2
        f2 = 0.7782 - 0.2746 * H_rib - 0.1437 * H_pf - 0.00668 * N_pf + 0.015559 * N_ac + 0.000176 * N_pf*N_pf + 0.1880 * H_rib*H_pf + 0.003317 * H_rib*N_pf - 0.009635 * H_rib*N_ac - 0.002637 * H_pf*N_pf - 0.003510 * H_pf*N_ac

        # 目标3
        f3 = 6.272 + 3.02 * H_rib + 6.08 * H_pf + 0.0368 * N_pf - 0.8848 * N_ac + 0.04381 * N_ac*N_ac + 6.35 * H_rib*H_pf - 0.3602 * H_rib*N_ac - 0.5497 * H_pf*N_ac


        out["F"] = [f1, f2, f3]

problem = My_problem()  




                                                                               

# 初始化算法     
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.mixed import MixedVariableGA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

algorithm = MixedVariableGA(pop_size=200,  # 选择群体大小
                            n_offsprings=100,  # 每代的后代个数
                            crossover=SBX(prob=0.5, eta=15), # prob 交叉概率 
                            mutation=PM(eta=20,), # 突变
                            eliminate_duplicate =True, # 启用重复检查(eliminate_duplicate =True)，确保交配产生的后代与它们自己和现有种群的设计空间值不同
                            survival=RankAndCrowdingSurvival())




# 设置终止条件
from pymoo.factory import get_termination
termination = get_termination("n_gen", 20)




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
# 将已缩放的X值还原为真实值
for i in X:
    i['H_rib'] = i['H_rib']/100
    i['H_pf'] = i['H_pf']/100
    i['N_ac'] = i['N_ac']*2
    


weight = [0.4,0.4,0.2] # 权重设置
NSGA2Post = post.NSGA2_post(X = X, F = F, history = res.history, HvDisplay='show') # 初始化函数
NSGA2Post.Compromise_Programming(weight) # 均衡规划法(Compromise Programming)
NSGA2Post.Pseudo_Weights(weight) # 伪权向量法(Pseudo-Weights)
date_hv = NSGA2Post.Hypervolume() # 进行超立方体积计算
date_Topsis = NSGA2Post.Topsis() # 进行TOPSIS评分
NSGA2Post.export_excel(date_hv,date_Topsis) # 将分析完成的列表导出为excel表格
NSGA2Post.export_Discrete_hist_excel(5) # 将每隔"gen"代数进行导出种群数据，该公式需手动修改NSGA2Post中对应函数
NSGA2Post.ParetoDisplay()



# from pymoo.util.reference_direction import UniformReferenceDirectionFactory
# from pymoo.visualization.pcp import PCP
# plot = PCP()
# plot.set_axis_style(color="grey", alpha=0.5)
# plot.add(F, color="grey", alpha=0.3)
# plot.add(F[40], linewidth=5, color="red")
# plot.add(F[12], linewidth=5, color="blue")
# plot.show()