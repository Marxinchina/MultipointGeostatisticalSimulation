import numpy as np
import matplotlib.pyplot as plt
import copy

'''
针对mps00的修正和优化，加入节点群多点直接采样算法
已知bug：搜索窗口截断ti是错误的，应该是ti被拓展
'''


class BunchDS:
    def __init__(self, ti_path, use_condition, node_num, scan_frac, threshold, simul_size,
                 cond_data=None, sim_random_path=True, scan_random_path=True, bunch_size=1):
        self.ti = np.loadtxt(ti_path)  # 获得格式正确的训练图像数据
        self.simul_size = simul_size  # 模拟区域
        # print(self.ti)
        self.cond_mode = use_condition  # 是否条件模拟， boolean类型
        self.node_num = node_num  # 数据模板预设的节点数
        self.scan_frac = scan_frac  # 最大扫描分数,最大扫描数与训练图像节点总数的比值
        self.threshold = threshold  # 结束模拟的距离阈值
        if self.cond_mode:
            self.cond_data = np.loadtxt(cond_data)  # 获得格式正确的硬数据
        else:
            self.cond_data = np.array([-9999, -9999, -9999, 0])
        # print(self.cond_data_path)
        self.sim_random_path = sim_random_path  # 模拟节点的模式：False.顺序，Ture.随机
        self.scan_random_path = scan_random_path  # 扫描图像的模式：False.顺序，Ture.随机
        self.bunch_size = bunch_size  # 节点群的尺寸 界定了节点数，计算公式为(bunch_size*2+1)**self.dim
        self.node_count = self.simul_size[0] * self.simul_size[1] * self.simul_size[2]  # 节点总数
        self.col = self.ti.shape[1]  # 训练数据的列数
        self.ti_size = [int(self.ti[-1, 0] - self.ti[0, 0] + 1), int(self.ti[-1, 1] - self.ti[0, 1] + 1),
                        int(self.ti[-1, 2] - self.ti[0, 2] + 1)]  # 训练图像的尺寸
        self.dim = self.get_dim()  # 判定模拟区域的维度
        self.ti_mat = self.ti_format()  # 矩阵型的训练图像数据
        # print(self.ti_mat)
        self.cond_data_mat = self.cond_format()[0:self.simul_size[0], 0:self.simul_size[1]]  # 矩阵型的条件数据，受到模拟区域的限制
        self.cond_data = self.cond_reformat()  # 重设条件数据
        # 备份条件数据
        cond_data_backup, cond_data_mat_backup = self.cond_data, self.cond_data_mat
        self.cond_data_backup = copy.deepcopy(cond_data_backup)
        self.cond_data_mat_backup = copy.deepcopy(cond_data_mat_backup)
        self.cond_data_mat_backup = self.cond_data_mat
        self.ti_mat_expand = self.ti_expand()  # 拓展后的训练图像，方便定义搜索窗口
        self.ti_ex = self.reformat(self.ti_mat_expand)  # 拓展后的训练图像，便于提取节点坐标
        self.xshift = int(self.ti_mat_expand.shape[0] / 3)  # x坐标自左向右的位移
        self.yshift = int(self.ti_mat_expand.shape[1] / 3)  # y坐标自上向下的位移
        self.cond_node_num = self.cond_data.shape[0]  # 条件数据个数
        self.count = 0  # 节点群不录用的次数
        self.facies = self.get_facies()  # 相数

    def get_facies(self):
        s = set(self.ti[..., -1])
        return len(s)

    def __str__(self):
        return '训练图像:\n{}\n硬数据:\n{}\n训练图像尺寸：\n{}\n模板节点数：\n{}\n模拟路径随机：' \
               '\n{}\n扫描路径随机：\n{}\n阈值：\n{}\n模拟范围：\n{}\n模拟维度：\n{}'.format(
            self.ti, self.cond_data, self.ti_size, self.node_num, self.sim_random_path,
            self.scan_random_path, self.threshold, self.simul_size, self.dim)

    # 矩阵型数据格式化为线性数据,不包含nan数据
    def cond_reformat(self):
        mat = self.cond_data_mat
        mat = self.reformat(mat)
        mat_id = np.isfinite(mat[..., -1])
        # print(mat[mat_id, ...].shape)
        return mat[mat_id, ...]

    def ti_format(self):  # 训练图像数据矩阵化
        return self.data_format(self.ti)

    def cond_format(self):  # 条件数据句矩阵化
        return self.data_format(self.cond_data)

    def data_format(self, mat):  # 数据集中格式化
        max_cr = self.ti_size
        if self.dim == 2:  # 二维情况
            if max_cr[0] == 1:  # x轴为扁平的第三维
                return self.mat_data(mat, 2, 1)
            if max_cr[1] == 1:  # y轴为扁平的第三维
                return self.mat_data(mat, 0, 2)
            if max_cr[2] == 1:  # z轴为扁平的第三维
                return self.mat_data(mat, 0, 1)
        else:
            return []

    def mat_data(self, mat, x, y):
        ts = self.ti_size
        res = np.full((ts[x], ts[y]), np.nan)  # 确保转化后的矩阵同训练图像一样大
        ms = mat.shape
        for i in range(int(ms[0])):
            res[int(mat[i, y]) - 1, int(mat[i, x]) - 1] = mat[i, -1]  # 行是y，列是x
        return res

    def get_dim(self):
        a = self.ti_size
        if a[0] == 1 or a[1] == 1 or a[2] == 1:  # 有一维没有长度
            return 2
        else:
            return 3

    def get_path(self, mode):
        if mode:  # 随机路径
            return np.random.permutation(self.node_count)
        else:  # 顺序路径
            return np.arange(self.node_count)

    # 为了应付一切情况，上下左右都拓展对应轴上的长度
    def ti_expand(self):
        ti = self.ti_mat
        left = right = np.zeros(ti.shape)
        temp = np.hstack((left, ti, right))
        top = bottom = np.zeros(temp.shape)
        res = np.vstack((top, temp, bottom))
        return res

    # 拓展训练图像数据的坐标需要调整
    def active_ti(self, xshift, yshift):
        ti = self.ti
        if self.ti_size[0] == 1:
            return ti + [0, yshift, xshift, 0]
        if self.ti_size[1] == 1:
            return ti + [xshift, 0, yshift, 0]
        if self.ti_size[2] == 1:
            return ti + [xshift, yshift, 0, 0]

    # 矩阵型数据格式化为线性数据,包含nan数据
    def reformat(self, mat):
        ms = mat.shape
        res = np.full((ms[0] * ms[1], 4), np.nan)
        for i in range(ms[0]):
            for j in range(ms[1]):
                if self.ti_size[0] == 1:
                    res[j + i * ms[0], ...] = [1, i + 1, j + 1, mat[i, j]]
                if self.ti_size[1] == 1:
                    res[j + i * ms[0], ...] = [j + 1, 1, i + 1, mat[i, j]]
                if self.ti_size[2] == 1:
                    res[j + i * ms[0], ...] = [j + 1, i + 1, 1, mat[i, j]]
        return res

    # 获得节点群
    def get_bunch(self, cp, simul):
        bunch_node_num = (self.bunch_size * 2 + 1) ** 2
        if bunch_node_num == 1:  # self.bunch_size 为0时就是原始直接采样法
            return cp - cp
        bunch_count = 0  # 节点群节点计数器
        bunch_template = np.zeros((bunch_node_num, 4))
        for i in range(-self.bunch_size, self.bunch_size + 1):
            for j in range(-self.bunch_size, self.bunch_size + 1):  # 列数是x轴
                p_x, p_y = int(cp[0]) + j, int(cp[1]) + i
                if self.ti_size[0] == 1:
                    pass
                if self.ti_size[1] == 1:
                    pass
                if self.ti_size[2] == 1:
                    # 在模拟网格上且节点值未知
                    if (0 < p_x <= self.simul_size[0] and 0 < p_y <= self.simul_size[1]) and np.isnan(
                            simul[p_y - 1, p_x - 1]):
                        bunch_template[bunch_count] = np.array([j, i, 0, 0])  # 要的是相对坐标
                        bunch_count += 1
                    else:
                        self.count += 1
                        # print('不在节点群中的是{}'.format([p_x, p_y, 1]))
        # if bunch_node_num - bunch_count != 0:
        #     print(bunch_node_num - bunch_count)
        return bunch_template[0:bunch_count]

    # 通过距离确定数据模板
    def get_event(self, cp, cond_data):
        n = self.node_num
        cp[-1] = 0
        if n > self.cond_node_num:
            return cond_data - cp
        else:
            dist = self.mdist(cp, cond_data)
            # print(dist)
            min_dist_idx = np.argsort(dist)
            # print(min_dist_idx)
            res = (cond_data - cp)[min_dist_idx, ...]
            # print(res[0:n, ...].shape)
            return res[0:n, ...]

    # 计算曼哈顿距离
    @staticmethod
    def mdist(mat1, mat2):
        res = np.abs(mat2 - mat1)
        # print(res)
        return np.sum(res[..., 0:-3], axis=1)

    # 输入坐标找到ti值
    def get_val(self, cps):
        # 输入的节点可能是一个也可能是多个,一个点不会出ti范围，但多个点不一定
        if len(cps.shape) == 1:
            if self.ti_size[0] == 1:
                cps[-1] = self.ti_mat[int(cps[1]), int(cps[2])]
            if self.ti_size[1] == 1:
                cps[-1] = self.ti_mat[int(cps[2]), int(cps[0])]
            if self.ti_size[2] == 1:
                cps[-1] = self.ti_mat[int(cps[1]), int(cps[0])]
        else:
            if self.ti_size[0] == 1:
                for ind in range(len(cps)):
                    cps[ind, -1] = self.ti_mat_expand[
                        int(cps[ind, 1]) - 1 + self.yshift, int(cps[ind, 2] - 1 + self.xshift)]
            if self.ti_size[1] == 1:
                for ind in range(len(cps)):
                    cps[ind, -1] = self.ti_mat_expand[
                        int(cps[ind, 2]) - 1 + self.yshift, int(cps[ind, 0] - 1 + self.xshift)]
            if self.ti_size[2] == 1:
                for ind in range(len(cps)):
                    # print(cps[ind, 0], int(cps[ind, 1]))
                    cps[ind, -1] = self.ti_mat_expand[
                        int(cps[ind, 1]) - 1 + self.yshift, int(cps[ind, 0] - 1 + self.xshift)]
        return cps

    # 模拟器分流
    def sim(self):
        if self.dim == 2:
            self.sim2()
        elif self.dim == 3:
            self.sim3()
        else:
            print('错误的数据格式')

    def sim3(self):
        pass

    # 展示image
    def show_data(self):
        if self.cond_mode:
            plt.subplot(1, 2, 1)
            # print(self.ti_mat)
            # print(self.cond_data_mat)
            plt.imshow(self.cond_data_mat)
            plt.subplot(1, 2, 2)
        plt.imshow(self.ti_mat)
        plt.show()

    def sim2(self):
        # 画出训练图像和硬数据
        # self.show_data()
        # pass
        # 初始化模拟网格
        initial_simul_mat = self.cond_data_mat
        # print(initial_simul_mat.shape)
        initial_simul = self.reformat(initial_simul_mat)
        # print(initial_simul)
        # plt.imshow(initial_simul_mat)
        # plt.show()

        # 获取路径,用同一个路径生成函数即可
        sim_path = self.get_path(self.sim_random_path)
        # scan_path = self.get_path(self.scan_random_path)
        # print(scan_path)
        # print(sim_path)

        # 扩展训练图像
        # ti_mat = self.ti_mat_expand
        # ti = self.ti_ex
        # print(ti)
        # plt.imshow(ti)
        # plt.show()
        # xshift = int(ti.shape[0] / 3)  # x坐标自左向右的位移
        # yshift = int(ti.shape[1] / 3)  # y坐标自上向下的位移
        # print(xshift)

        # 生成节点群框架
        # bunch_node_num = (self.bunch_size * 2 + 1) ** 2  # self.bunch_size 为0时就是原始直接采样法
        # bunch_template = np.zeros((bunch_node_num, 4))
        # print(bunch_template)

        # 开始对每一个节点模拟
        # print(best_point_val)
        simul = initial_simul  # 还没开始模拟
        simul_mat = initial_simul_mat  # 只有值数据
        # plt.imshow(simul_mat)
        # plt.show()
        active = self.active_ti(self.xshift, self.yshift)  # 移动了坐标轴
        sim_node_count = 0  # 模拟节点计数器
        # print(active)
        for sim in sim_path:
            sim_point = simul[sim]  # 坐标数据和值数据

            sim_node_count += 1  # 计数器自增
            print('正在模拟第{:>8d}个节点'.format(sim_node_count), end='\t')

            # print(sim_point)
            sim_bunch = self.get_bunch(sim_point, simul_mat)  # 节点群的相对位置
            # print(sim_bunch)
            bunch_node_num = len(sim_bunch)
            if not np.isnan(sim_point[-1]):  # 不模拟硬数据
                for ind in range(bunch_node_num):
                    if np.sum(sim_bunch[ind, 0:3]) == 0:
                        break
                sim_bunch = np.delete(sim_bunch, ind)
                bunch_node_num -= 1

            # print(sim_bunch)
            # print(self.count)


            node_event = self.get_event(sim_point, self.cond_data)  # 获得节点的数据事件，采用的相对模拟中心节点的相对坐标
            event_template = node_event
            event_template[..., -1] = np.zeros(event_template[..., -1].shape)  # 整理出数据事件模板
            # print(event_template)

            min_dist = 10000  # 初始化最小距离
            best_point_val = np.zeros((bunch_node_num, 1))
            scan_count = 0  # 扫描节点计数器
            scan_bunch_point = sim_bunch  # 初始化扫描节点的节点群
            # 开始扫描训练图像的节点
            scan_path = self.get_path(self.scan_random_path)  # 设定扫描路径
            for scan in scan_path:
                scan_point = self.ti[scan]
                # print(scan_point)
                # print(self.ti)
                # print(scan_point)

                # 查找相关节点群
                scan_bunch_point = scan_point + sim_bunch  # scan_point坐标多了1
                # print(scan_bunch_point)
                # 为节点群找到值
                scan_bunch_point_val = self.get_val(scan_bunch_point)
                # print(scan_bunch_point_val)

                # 查找数据事件
                scan_event = scan_point + event_template
                # # 为数据事件找到值
                scan_event_val = self.get_val(scan_event)
                # print(scan_event_val)

                dist = self.dist(node_event[..., -1], scan_event_val[..., -1])

                if dist == 0:
                    best_point_val = scan_bunch_point_val[..., -1]
                    break

                if dist <= self.threshold:
                    best_point_val = scan_bunch_point_val[..., -1]
                    # print('达到阈值了')
                    break

                if dist < min_dist:
                    min_dist = dist
                    best_point_val = scan_bunch_point_val[..., -1]

                scan_count += 1
                if scan_count >= self.scan_frac * self.node_count:
                    best_point_val = scan_bunch_point_val[..., -1]
                    break

            # 更新匹配的数据到模拟网格


        # 至此已经模拟结束
        # plt.subplot(1, 3, 1)
        # plt.imshow(self.ti_mat)
        # plt.title('TI')
        #
        # plt.subplot(1, 3, 2)
        # plt.imshow(self.cond_data_mat_backup)
        # plt.title('cond_data_path')
        #
        # plt.subplot(1, 3, 3)
        # plt.imshow(self.cond_data_mat)
        # plt.title('sim')
        #
        # plt.show()

        # print(self.count)



    def dist(self, mat1, mat2):
        diff = np.abs(mat1 - mat2)
        if self.facies > 10:
            # 差值归一化
            d_max = np.max(diff)
            return np.sum(diff) / (d_max * len(diff))
        else:
            # 曼哈顿距离
            return np.mean(np.where(diff != 0))


if __name__ == '__main__':
    ti = 'TI_A.txt'
    cond = 'Cond_A.txt'
    ti_path = ti
    use_condition = True
    node_num = 30
    scan_frac = 0.01
    threshold = 0.01
    cond_data = cond
    sim_random_path = True
    scan_random_path = True
    bunch_size = 1
    simul_size = [64, 64, 1]

    mps = BunchDS(ti_path, use_condition, node_num, scan_frac, threshold, simul_size
                  , cond_data, sim_random_path, scan_random_path,
                  bunch_size)
    # print(mps)
    mps.sim()
