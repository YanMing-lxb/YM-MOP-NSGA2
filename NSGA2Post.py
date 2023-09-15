import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 乌托邦点最小距离

class NSGA2_post(object):
    def __init__(self,
                 X = None,
                 F = None,
                 history = None,
                 HvDisplay=None):
        self.history = history
        self.X = X
        self.F = F
        self.Fl = F.min(axis=0) # 取所有f的最小值
        self.Fu = F.max(axis=0) # 取所有f的最大值

        self.nF = (self.F - self.Fl) / (self.Fu - self.Fl) # 归一化计算
        self.nFl = self.nF.min(axis=0) # 归一化后取所有f的最小值
        self.nFu = self.nF.max(axis=0) # 归一化后取所有f的最大值
        self.HvDisplay = HvDisplay
        print("关键点: \n乌托邦解：%s\n最差解： %s\n" % (self.Fl, self.Fu))
    
# 多标准决策
    # 均衡规划法(Compromise Programming)
    def Compromise_Programming(self,weight_list):
        from pymoo.decomposition.asf import ASF
        weights = np.array(weight_list) # 比重
        decomp = ASF()
        global I
        I = decomp(self.nF, 1/weights).argmin()
        print("均衡规划法解值: \n序号：%s\n响应值： %s\n参数值： %s\n " % (I, self.F[I], self.X[I]))
        
    # 伪权向量法(Pseudo-Weights)
    def Pseudo_Weights(self, weight_list):
        
        from pymoo.mcdm.pseudo_weights import PseudoWeights
        weights = np.array(weight_list) # 比重
        global i
        i = PseudoWeights(weights).do(self.nF)
        print("伪权重向量法解值: \n序号：%s\n响应值： %s\n参数值： %s\n " % (i, self.F[i],self.X[i]))

    # TOPSIS算法决策
    def Topsis(self):
        # 正向化处理
        SF = self.Fu-self.F 

        #正向化矩阵标准化(去除量纲影响)
        def Standard(datas):
            k = np.power(np.sum(pow(datas,2) , axis = 0) , 0.5)
            for i in range(len(k)):
                datas[:,i] = datas[:,i] / k[i]
            return datas

        sta_F=Standard(SF)

        #计算得分并归一化
        def Score(sta_data):
            z_max = np.amax(sta_data,axis=0)
            z_min = np.amin(sta_data,axis=0)
            #计算每一个样本点与最大值的距离
            tmpmaxdist = np.power(np.sum(np.power((z_max - sta_data) , 2) , axis = 1) , 0.5)
            tmpmindist = np.power(np.sum(np.power((z_min - sta_data) , 2) , axis = 1) , 0.5)
            score = tmpmindist / (tmpmindist + tmpmaxdist)
            score = score / np.sum(score)  # 归一化处理;
            # score = np.around(score, decimals=5) # 保留精度为3
            return score
        
        sco = Score(sta_F)   #计算得分
        sco_max = sco.max(axis=0) # 最高分
        global sco_index
        sco_index = sco.tolist().index(sco_max) # 获取最高值的序号
        print("TOPSIS算法解值: \n序号：%s\n响应值： %s\n参数值： %s\nscore max： %s\n " % (sco_index, self.F[sco_index],self.X[sco_index],sco_max))
        
        return sco


    # 收敛性分析 
    def Hypervolume(self):
        hist = self.history
        # print(len(hist)) # 40
        n_evals = []             # 函数评估次数
        hist_F = []              # 每次迭代的目标值
        hist_cv = []             # 每次迭代的约束违反
        hist_cv_avg = []         # 所有种群的平均约束违反

        for algo in hist:

            # 存储评估数
            n_evals.append(algo.evaluator.n_eval)
            # 从算法中检索最优
            opt = algo.opt

            # 存储约束违反
            hist_cv.append(opt.get("CV").min())
            hist_cv_avg.append(algo.pop.get("CV").mean())

            # 过滤出可行解，并添加至目标空间
            feas = np.where(opt.get("feasible"))[0]
            hist_F.append(opt.get("F")[feas])

            # Hypervolume 算法评估
            from pymoo.indicators.hv import Hypervolume
            metric = Hypervolume(ref_point= np.array(self.nFu),
                                norm_ref_point=False,
                                zero_to_one=True,
                                ideal=self.Fl,
                                nadir=self.Fu)
            hv = [metric.do(_F) for _F in hist_F]
        
        # 可视化Hypervolume
        if self.HvDisplay == 'show':
            plt.figure(figsize=(7, 5))
            plt.plot(n_evals, hv,  color='black', lw=0.7, label="Avg. CV of Pop")
            plt.scatter(n_evals, hv,  facecolor="none", edgecolor='black', marker="p")
            plt.title("Convergence")
            plt.xlabel("Function Evaluations")
            plt.ylabel("Hypervolume")
            plt.show()
        else:
            print("如需可视化请在NSGA2_post中输入输入HvDisplay='show'\n")
        return n_evals,hv


    # 结果写入excel
    
    def export_excel(self,date_hv,date_Topsis):
        #指定生成的Excel表格名称
        file_path = pd.ExcelWriter('NSGA-Ⅱ优化结果.xlsx')
        #输出
        date1 = pd.DataFrame(list(self.X)) # 写入所有变量
        date1.columns = ["X"+str(i) for i in range(1, len(date1.columns)+1)]
        date2 = pd.DataFrame(list(self.F)) # 写入所有响应值
        date2.columns = ["F"+str(i) for i in range(1, len(date2.columns)+1)]
        date3 = pd.DataFrame(list(self.nF)) # 写入所有归一化后的响应值
        date3.columns = ["归一F"+str(i) for i in range(1, len(date3.columns)+1)]
        date4 = pd.DataFrame({"Function Evaluations": date_hv[0], "Hypervolume": date_hv[1]}) # Hypervolume的值
        date5 = pd.DataFrame({"TOPSIS score": date_Topsis}).applymap(lambda x: '%.9f'%x) # 写入 Topsis 评分
        date_merge = pd.concat([date1,date2,date3,date5,date4],axis = 1) # 进行横向合并datafram 
        date_merge.to_excel(file_path, '优化结果', float_format='%.5f') # 写入合并后的数据
        
        # 保存文件
        file_path.close() 
    
    def export_hist_excel(self, gen): 
        #指定生成的Excel表格名称
        file_path = pd.ExcelWriter('NSGA-Ⅱ过程数据.xlsx')
        # 提取NSGA-Ⅱ繁衍过程中种群数据
        for e in range(len(self.history)):
            if  e%gen == 0: # 每隔 "gen" 代提一次数据
                valF = self.history[e].pop.get("F") # off 为子代，opt为筛选后的所有解，pop为优化前的所有解
                valX = self.history[e].pop.get("X") # 从历史数据中提取X值
                date6 = pd.DataFrame(list(valX)) # 写入所有变量
                date6.columns = ["X"+str(i) for i in range(1, len(date6.columns)+1)]
                date7 = pd.DataFrame(list(valF)) # 写入响应数据
                date7.columns = ["F"+str(i) for i in range(1, len(date7.columns)+1)]
                #指定字段顺序
                date_merge = pd.concat([date6,date7],axis = 1)
                date_merge.to_excel(file_path, '第'+str(e)+'代', float_format='%.5f')
        # 保存文件
        file_path.close() 

    def export_Discrete_hist_excel(self, gen): 
        #指定生成的Excel表格名称
        file_path = pd.ExcelWriter('NSGA-Ⅱ过程数据.xlsx')
        # 提取NSGA-Ⅱ繁衍过程中种群数据
        for e in range(len(self.history)):
            if  e%gen == 0: # 每隔 "gen" 代提一次数据
                valF = self.history[e].pop.get("F") # off 为子代，opt为筛选后的所有解，pop为优化前的所有解
                valX = self.history[e].pop.get("X") # 从历史数据中提取X值
                # for i in valX:
                #     # 以下公式根据需求修改
                #     i['H_rib'] = i['H_rib']/100
                #     i['H_pf'] = i['H_pf']/100
                #     i['N_ac'] = i['N_ac']*2
                #     # 以上公式根据需求修改
                date6 = pd.DataFrame(list(valX)) # 写入所有变量
                date6.columns = ["X"+str(i) for i in range(1, len(date6.columns)+1)]
                date7 = pd.DataFrame(list(valF)) # 写入响应数据
                date7.columns = ["F"+str(i) for i in range(1, len(date7.columns)+1)]
                #指定字段顺序
                date_merge = pd.concat([date6,date7],axis = 1)
                date_merge.to_excel(file_path, '第'+str(e)+'代', float_format='%.5f')
        # 保存文件
        file_path.save() 
        
    def ParetoDisplay(self):
        DisplayType = len(self.F[0])
        if DisplayType == 2:
            approx_ideal = self.Fl
            approx_nadir = self.Fu
            plt.figure(figsize=(7, 5))
            print(self.F[I])
            print(self.F[I][0])
            plt.scatter(self.F[:, 0], self.F[:, 1], s=30, facecolors='none', edgecolors='k')
            plt.scatter(self.F[I][0], self.F[I][1], color="red",marker="o", s=50, label="Compromise Programming") # 显示均衡规划的最佳点
            plt.scatter(self.F[i][0], self.F[i][1], color="green",marker="^", s=50, label="Pseudo-Weights") # 显示伪权向量的最佳点
            plt.scatter(self.F[sco_index][0], self.F[sco_index][1], color="blue",marker="X", s=50, label="TOPSIS") # 显示TOPSIS算法解值
            plt.scatter(approx_ideal[0], approx_ideal[1], facecolors='none', edgecolors='red', marker="*", s=100, label="Ideal Point") # 显示乌托邦解
            plt.scatter(approx_nadir[0], approx_nadir[1], facecolors='none', edgecolors='black', marker="p", s=100, label="Nadir Point") # 显示最差点
            plt.title("Objective Space")
            plt.legend()
            plt.show() 
            
        elif DisplayType == 3:
            from pymoo.visualization.scatter import Scatter
            plot = Scatter(legend=True)
            plot.add(self.F, facecolor="none", edgecolor="k", alpha=0.5, s=10) # 显示所有解
            plot.add(self.F[I], color="red",marker="o", s=50, label="Compromise Programming") # 显示均衡规划的最佳点
            plot.add(self.F[i], color="green",marker="^", s=50, label="Pseudo-Weights") # 显示伪权向量的最佳点
            plot.add(self.F[sco_index], color="blue",marker="X", s=50, label="TOPSIS") # 显示TOPSIS算法解值
            plot.add(self.Fl, facecolors='none', edgecolors='red', marker="*", s=50, label="Ideal Point") # 显示乌托邦解
            plot.add(self.Fu, facecolors='none', edgecolors='black', marker="p", s=50, label="Nadir Point") # 显示最差点
            plot.show()
        
        '''


# 可视化



'''