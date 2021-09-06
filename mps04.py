import numpy as np
import matplotlib.pyplot as plt

'''
针对mps03的优化，
已知问题：无法循环模拟
针对性优化：实时更新待模拟节点和模拟路径

'''
plt.rcParams['font.sans-serif'] = ['SimHei']  # 防止中文标签乱码，还有通过导入字体文件的方法
plt.rcParams['axes.unicode_minus'] = False


class BunchDS:
    def __init__(self, ti_path, node_num, scan_frac, threshold, simul_size,
                 cond_data_path, sim_random_path=True, scan_random_path=True, bunch_size=1, show_mode=True):
        self.ti_path = ti_path
        self.node_num = node_num
        self.scan_frac = scan_frac
        self.threshold = threshold  # 扫描终止阈值
        self.simul_size = simul_size  # 模拟范围
        self.cond_data_path = cond_data_path
        self.sim_random_path = sim_random_path
        self.scan_random_path = scan_random_path
        self.bunch_size = bunch_size  # 节点群的尺寸 界定了节点数，计算公式为(bunch_size*2+1)**self.dim

        self.ti = np.loadtxt(self.ti_path)  # 训练图像
        self.cond_data = np.loadtxt(self.cond_data_path)  # 条件数据
        self.ti_size = [int(np.max(self.ti[..., 0]) - np.min(self.ti[..., 0]) + 1),
                        int(np.max(self.ti[..., 1]) - np.min(self.ti[..., 1]) + 1),
                        int(np.max(self.ti[..., 2]) - np.min(self.ti[..., 2]) + 1)]
        self.ti = self.format_ti()
        self.facies = len(set(self.ti[..., -1]))
        self.all_node = self.ti_size[0] * self.ti_size[1] * self.ti_size[2]
        self.max_scan_points = self.all_node * self.scan_frac
        self.show_mode = show_mode

    # 解决坐标z轴问题
    def format_ti(self):
        ti, ti_size = self.ti, self.ti_size
        ti1, ti2, ti3, ti4 = ti[..., 0], ti[..., 1], ti[..., 2], ti[..., 3]
        if self.ti_size[0] == 1:
            ti = np.vstack((ti3, ti2, ti1, ti4))
            ti_size[2], ti_size[0] = ti_size[0], ti_size[2]
        if self.ti_size[1] == 1:
            ti = np.vstack((ti1, ti3, ti2, ti4))
            ti_size[1], ti_size[2] = ti_size[2], ti_size[1]
        if self.ti_size[2] == 1:
            ti = np.vstack((ti1, ti2, ti3, ti4))
        self.ti_size = ti_size
        return ti.T

    def __str__(self):
        return 'TI_path:{}\nCond_data_path:{}\nTI_size:{}\nfacies:{}\nsimul_size:{}\nevent_node_num:{}\n' \
               'max_scan_point:{}'.format(self.ti_path, self.cond_data_path, self.ti_size,
                                          self.facies,
                                          self.simul_size,
                                          self.node_num, self.max_scan_points)

    def sim(self):
        if self.ti_size[2] == 1:
            return self.sim2()
        else:
            return self.sim3()

    def sim3(self):
        return 0

    @staticmethod
    def data2mat(data, size):
        res = np.full((int(size[1]), int(size[0])), np.nan)
        for m in data:
            res[int(m[1]) - 1, int(m[0]) - 1] = m[-1]
        return res

    @staticmethod
    def data1D2mat(data1D, size):
        res = np.full((int(size[1]), int(size[0])), np.nan)
        for i in range(size[1]):  # i是y轴是行
            for j in range(size[0]):  # j是x轴是列
                # print(data1D[j + i * size[0]])
                id = j + i * size[0]
                res[i, j] = data1D[0, id]
        return res

    @staticmethod
    def show_data(*args):
        num, count = len(args), 1
        plt.clf()
        for a in args:
            plt.subplot(1, num, count)
            plt.imshow(a)
            count += 1
        plt.show()

    def get_path(self, random_path, all_node):
        if random_path:
            return np.random.permutation(all_node)
        else:
            return np.arange(all_node)

    @staticmethod
    def mat2data(mat):
        size = mat.shape
        row, col, l = 1, 1, size[0] * size[1]
        res = np.full((l, 4), np.nan)
        for i in range(l):
            res[i, 0], res[i, 1], res[i, 2], res[i, 3] = col, row, 1, mat[row - 1, col - 1]
            col += 1
            if col > size[0]:
                col = 1
                row += 1
        return res

    def get_bunch(self, sim_p, sg):
        sim_px, sim_py = int(sim_p[0]), int(sim_p[1])
        size = self.bunch_size
        bunch_num = (2 * size + 1) ** 2
        bunch_points = np.zeros((bunch_num, 4))
        # print(bunch_points)
        bunch_count = 0
        for i in range(-size, size + 1):
            for j in range(-size, size + 1):
                px, py = (j + sim_px), (i + sim_py)
                # print(px, py, sg[py-1, px-1], sep=' ')
                # 从中心节点左上角开始的节点需要依次判定是否在模拟网格内（包含边界），以及是否有值而不用模拟
                if (0 < px <= self.simul_size[0] and 0 < py <= self.simul_size[1]) and np.isnan(sg[py - 1, px - 1]):
                    bunch_points[bunch_count, 0], bunch_points[bunch_count, 1], bunch_points[
                        bunch_count, 2], bunch_points[bunch_count, 3] = j, i, 0, 0

                    bunch_count += 1
        return bunch_points[0:bunch_count]

    def get_data_event(self, sim_p, simul):
        number = 1 if len(simul.shape) == 1 else len(simul)
        if number < self.node_num:
            return simul
        else:
            # 先计算节点到各个条件数据的距离，在截取最近的节点
            dist = self.mdist(sim_p, simul)
            idx = np.argsort(dist)[0:self.node_num]
            return simul[idx]

    @staticmethod
    def mdist(mat1, mat2):
        diff = np.abs(mat2 - mat1)
        return np.sum(diff[..., 0:3], axis=1)

    @staticmethod
    def no_nan(mat):
        idx = np.isfinite(mat[..., -1])
        # print(idx)
        return mat[idx]

    # 拓展为搜索窗口
    @staticmethod
    def ti2search_window(mat):
        left = right = np.full(mat.shape, np.nan)
        temp = np.hstack((left, mat, right))
        top = bottom = np.full(temp.shape, np.nan)
        res = np.vstack((top, temp, bottom))
        return res

    # 通过坐标获取值
    def get_val(self, points, search_window, xshift, yshift):
        if len(points.shape) == 1:
            l = 1
            res = np.zeros((l, 4))

            res[0], res[1], res[2], res[3] = points[0], points[1], points[2], search_window[
                int(points[1] + yshift - 1), int(points[0] + xshift - 1)]
        else:
            l = len(points)
            res = np.zeros((l, 4))
            for i in range(l):
                res[i, 0], res[i, 1], res[i, 2], res[i, 3] = points[i, 0], points[i, 1], points[i, 2], search_window[
                    int(points[i, 1] + yshift - 1), int(points[i, 0] + xshift - 1)]
        return res

    def dist(self, mat1, mat2):
        diff = abs(mat1 - mat2)
        if self.facies > 10:
            d_max = np.max(diff)
            return np.sum(diff) / (d_max * len(diff))
        else:
            return np.mean(diff != 0)

    @staticmethod
    def has_nan(mat):
        idx = np.isnan(mat[..., -1])
        # print(idx)
        return mat[idx]

    def sim2(self):
        # 展示ti与cond_data
        ti = self.ti
        cond_data = self.cond_data
        # 先将它们转换称可作图矩阵
        ti_mat = self.data2mat(ti, self.ti_size)
        # 限定模拟范围
        cond_data_mat = self.data2mat(cond_data, self.ti_size)[:self.simul_size[1], :self.simul_size[0]]
        if self.show_mode:
            self.show_data(ti_mat, cond_data_mat)
        # print(cond_data)

        # 获取一个拓展后的训练图像作为搜索窗口
        search_window_mat = self.ti2search_window(ti_mat)
        # self.show_data(search_window_mat)
        search_window = self.mat2data(search_window_mat)
        xshift = search_window_mat.shape[1] // 3
        yshift = search_window_mat.shape[0] // 3
        # max_scan_point = self.scan_frac * self.all_node
        sim_node_num = cond_data_mat.size
        iter_val = np.zeros((1, sim_node_num))  # 节点值存储器


        # 初始化模拟网格
        simul_grid_mat = cond_data_mat
        simul_grid = self.mat2data(simul_grid_mat)  # 模拟网格线性化
        # self.show_data(simul_grid_mat)
        # print(simul_grid_no_nan)
        # 模拟节点计数器
        sim_count = 0
        max_point_out = 0  # 最大扫描退出计数器
        zero_dist_out = 0  # 0距离退出计数器
        threshold_out = 0  # 小于阈值退出计数器
        un_need_sim_point = 0  # 不需要模拟周边及自身的节点计数器
        if self.show_mode:
            plt.ion()
        while True:
            simul_grid_no_nan = self.no_nan(simul_grid)  # 更新到线性化模拟网格去nan
            simul_grid_nan = self.has_nan(simul_grid)  # 更新到线性化模拟网格保留nan
            # 生成除去条件数据的模拟路径
            nan_node = len(simul_grid_nan)
            sim_path = self.get_path(self.sim_random_path, nan_node)
            if nan_node != 0:
                sim_id = sim_path[np.random.randint(nan_node)]
            else:
                plt.savefig('test.png')
                break
            # print(simul_grid_no_nan.shape[0])
            sim_point = simul_grid_nan[sim_id]
            # print(sim_point)
            sim_count += 1
            print('正在模拟第{}个节点,该节点是{}'.format(sim_count, sim_point), end='\t')

            # 判断是否为硬数据
            # if not np.isnan(sim_point[-1]):
            #     print('条件数据和已经模拟过的数据不再模拟')
            #     continue

            # 设置节点群
            bunch_points = np.array([0, 0, 0, 0])
            if self.bunch_size != 0:
                bunch_points = self.get_bunch(sim_point, simul_grid_mat)  # 这只是相对位置
            bunch_num = 1 if len(bunch_points.shape) == 1 else len(bunch_points)
            # print(bunch_points)
            # print(bunch_num)

            # 设置数据事件
            data_event = self.get_data_event(sim_point, simul_grid_no_nan)
            # print(data_event)
            # 设置数据事件模板
            data_event_template = data_event - sim_point
            data_event_template[..., -1] = 0
            # print(data_event_template)

            # 设置扫描路径
            scan_path = self.get_path(self.scan_random_path, self.all_node)
            scan_count = 0
            mindist = 1  # 距离[0,1]
            best_point_val = np.full((bunch_num, 1), np.nan)
            best_point_x = 0
            best_point_y = 0
            # 开始逐点扫描
            for scan_id in scan_path:
                scan_point = ti[scan_id]
                scan_count += 1
                # 确定该节点的数据事件
                scan_data_event = data_event_template + scan_point
                scan_data_event = self.get_val(scan_data_event, search_window_mat, xshift, yshift)
                # print(scan_data_event)

                # 计算扫描节点与模拟节点数据事件之间的相似度
                dist = self.dist(scan_data_event[..., -1], data_event[..., -1])

                # 找到节点群的值
                scan_points_bunch = scan_point + bunch_points
                scan_points_bunch = self.get_val(scan_points_bunch, search_window_mat, xshift, yshift)
                # print(scan_points_bunch)
                # x先更新最小距离
                if dist < mindist:
                    if len(bunch_points):
                        best_point_x = scan_point[0]
                        best_point_y = scan_point[1]
                    else:
                        best_point_x = -1
                        best_point_y = -1
                    best_point_val = scan_points_bunch[..., -1]
                    mindist = dist

                if dist <= self.threshold:
                    threshold_out += 1
                    best_point_val = scan_points_bunch[..., -1]
                    break

                if dist == 0:
                    zero_dist_out += 1
                    best_point_val = scan_points_bunch[..., -1]
                    break

                if scan_count >= np.floor(self.max_scan_points):
                    max_point_out += 1
                    break
            if best_point_x == -1 and best_point_y == -1:
                un_need_sim_point += 1
            # 将匹配到的节点群更新到模拟网格，节点群有可能只有一个点
            # print(bunch_points + sim_point)

            sim_points = bunch_points + sim_point
            # print(sim_points)
            # print(best_point_val)
            for i in range(len(best_point_val)):  # best_point_val是一个数组
                # print(simul_grid_mat.shape)
                y = int(sim_points[i, 1]) - 1
                x = int(sim_points[i, 0]) - 1
                # print(simul_grid_mat[y, x])
                simul_grid_mat[y, x] = best_point_val[i]

            # 更新到线性数据中
            simul_grid = self.mat2data(simul_grid_mat)

            sim_frac = np.round((100 * simul_grid_no_nan.shape[0]) / len(simul_grid), 2)
            # print(sim_frac)

            if self.show_mode:

                plt.clf()  # 清除刷新前的图表，防止数据量过大消耗内存
                plt.suptitle("实时更新图", fontsize=12)
                plt.subplot(1, 2, 1)
                plt.text(0, 0.5,
                         '正在模拟点({},{})及其周边{}个节点\n通过距离0完成模拟：{}\n达到最大模拟分数完成模拟：{}\n达到阈值退出模拟：'
                         '{}\n模拟完成的总结点数：{}\n匹配到的最佳点位为({},{})'.format(
                             sim_point[0], bunch_num, sim_point[1], zero_dist_out, max_point_out, threshold_out,
                             sim_count,
                             best_point_x,
                             best_point_y))

                plt.text(0, 0.2,
                         '本次训练图像共{}个未知点\n其中{}个点不用模拟\n每次扫描最多需扫描{}个点\n本次扫描了{}个节点\n当前模拟进度{}%'.format(
                             self.all_node - len(cond_data), un_need_sim_point, self.max_scan_points, scan_count,
                             sim_frac))
                # plt.gca().get_xaxis().set_visible(False)
                # plt.gca().get_yaxis().set_visible(False)
                plt.axis('off')
                plt.subplot(1, 2, 2)
                plt.imshow(simul_grid_mat)
                plt.pause(0.001)

            print('ok')
            if sim_frac >= 99.9999:  # 应对浮点数
                if self.show_mode:
                    plt.savefig('test.png')
                    plt.ioff()
                # plt.show()
                # plt.close()
                break

        # 更新节点值到存储器
        temp = simul_grid[..., -1]
        # print(temp)
        # print(iter_val)
        iter_val += temp

        # 按列取平均
        final_val = iter_val / it

        # 作图
        final_val_mat = self.data1D2mat(final_val, self.simul_size)
        return final_val_mat

    # 计算变异函数
    def semi_var(self):
        pass

if __name__ == '__main__':
    ti = 'TI_A.txt'
    cond = 'Cond_A.txt'
    ti_path = ti
    node_num = 30
    scan_frac = 0.5
    threshold = 0.01
    cond_data_path = cond
    sim_random_path = True
    scan_random_path = True
    bunch_size = 2
    simul_size = [64, 64, 1]
    # simul_size = [8, 8, 1]
    it = 1
    show_mode = True

    mps = BunchDS(ti_path, node_num, scan_frac, threshold, simul_size, cond_data_path, sim_random_path,
                  scan_random_path, bunch_size, show_mode)
    print(mps)
    res = 0
    for i in range(it):
        val = mps.sim()
        res += val
        print(val)
        print(res)
    res /= it
    plt.imshow(res)
    plt.show()


