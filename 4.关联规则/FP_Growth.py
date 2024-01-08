class Node:
    def __init__(self, node_name, count, parentNode):
        self.name = node_name
        self.count = count
        self.nodeLink = None  # 根据nideLink可以找到整棵树中所有nodename一样的节点
        self.parent = parentNode  # 父亲节点
        self.children = {}  # 子节点{节点名字:节点地址}


class Fp_growth():
    def update_header(self, node, targetNode):  # 更新headertable中的node节点形成的链表
        # 寻找到node里面最底层的nodeLink
        while node.nodeLink != None:
            node = node.nodeLink
        node.nodeLink = targetNode

    def update_fptree(self, items, node, headerTable):  # 用于更新fptree
        # 更新排序后的商品表中的第一个商品
        if items[0] in node.children:
            # 判断items的第一个结点是否已作为子结点，如果有，将统计值+1
            node.children[items[0]].count += 1
        else:
            # 如果没有，创建该孩子结点，并且记录在children里
            node.children[items[0]] = Node(items[0], 1, node)
            # 并且如果该name中不存在结点，将该结点记录在项头表里面
            if headerTable[items[0]][1] == None:
                headerTable[items[0]][1] = node.children[items[0]]
            # 如果该name中存在结点，为该结点建立关联关系（关联上新创建的这个结点）
            else:
                self.update_header(headerTable[items[0]][1], node.children[items[0]])
        # 递归
        if len(items) > 1:
            self.update_fptree(items[1:], node.children[items[0]], headerTable)

    def create_fptree(self, data_set, min_support, flag=False):  # 建树主函数
        '''
        根据data_set创建fp树
        header_table结构为
        {"nodename":[num,node],..} 根据node.nodelink可以找到整个树中的所有nodename
        '''
        # 1.计算相头表
        item_count = {}  # 统计各项出现次数
        for t in data_set:  # 第一次遍历，得到频繁一项集
            for item in t:
                if item not in item_count:
                    item_count[item] = 1
                else:
                    item_count[item] += 1
        headerTable = {}
        for k in item_count:  # 剔除不满足最小支持度的项
            if item_count[k] >= min_support:
                headerTable[k] = item_count[k]
        freqItemSet = set(headerTable.keys())  # 满足最小支持度的频繁项集
        if len(freqItemSet) == 0:
            return None, None
        # headerTable项头表的字典表示：{name:[count,node]}
        for k in headerTable:
            headerTable[k] = [headerTable[k], None]
        # 创建根结点，name，count，parent
        tree_header = Node('head node', 1, None)
        if flag:
            ite = tqdm(data_set)
        else:
            ite = data_set
        # 2.第二次遍历，创建FP Tree
        for t in ite:
            localD = {}
            # 过滤，只取该样本中满足最小支持度的频繁项，{name:count}
            for item in t:
                if item in freqItemSet:
                    localD[item] = headerTable[item][0]  # element : count
            if len(localD) > 0:
                # 根据count从大到小对单样本排序
                order_item = [v[0] for v in sorted(localD.items(), key=lambda x: x[1], reverse=True)]
                # 用过滤且排序后的样本更新树,对于结点记录父节点、子节点、统计次数、同名节点之间的链表
                self.update_fptree(order_item, tree_header, headerTable)
        return tree_header, headerTable

    def find_path(self, node, nodepath):
        '''
        递归将node的父节点添加到路径
        '''
        if node.parent != None:
            nodepath.append(node.parent.name)
            self.find_path(node.parent, nodepath)

    def find_cond_pattern_base(self, node_name, headerTable):
        '''
        根据项头表（或者子项头表）节点名字，找出所有条件模式基
        '''
        treeNode = headerTable[node_name][1]
        cond_pat_base = {}  # 保存所有条件模式基
        while treeNode != None:
            nodepath = []
            self.find_path(treeNode, nodepath)  # 寻找到该结点的路径
            if len(nodepath) > 1:
                cond_pat_base[frozenset(nodepath[:-1])] = treeNode.count
            treeNode = treeNode.nodeLink
        return cond_pat_base

    def create_cond_fptree(self, headerTable, min_support, temp, freq_items, support_data):
        '''构建条件模式子树
        :param headerTable:项头表
        :param min_support: 最小支持度
        :param temp:当前条件模式子树对应的name set
        :param freq_items:记录所有满足支持度的频繁项集
        :param support_data:每个项集频繁项集的支持度
        :return:
        '''
        # 1.根据出现的频数，对项头表的name进行排序
        freqs = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])]  # 根据频繁项的总频次排序
        for freq in freqs:  # 对每个频繁项
            freq_set = temp.copy()
            freq_set.add(freq)
            freq_items.add(frozenset(freq_set))
            if frozenset(freq_set) not in support_data:  # 检查该频繁项是否在support_data中
                support_data[frozenset(freq_set)] = headerTable[freq][0]
            else:
                support_data[frozenset(freq_set)] += headerTable[freq][0]

            cond_pat_base = self.find_cond_pattern_base(freq, headerTable)  # 寻找到所有条件模式基
            # 将条件模式基字典{["A","B"]:2}转化为数组[["A","B"],["A","B"]]
            cond_pat_dataset = []
            for item in cond_pat_base:
                item_temp = list(item)
                item_temp.sort()
                for i in range(cond_pat_base[item]):
                    cond_pat_dataset.append(item_temp)
            # 创建条件模式子树
            cond_tree, cur_headtable = self.create_fptree(cond_pat_dataset, min_support)
            if cur_headtable != None:
                self.create_cond_fptree(cur_headtable, min_support, freq_set, freq_items, support_data)  # 递归挖掘条件FP树

    def rule_mining(self, data_set, min_support):
        freqItemSet = set()
        support_data = {}
        tree_header, headerTable = self.create_fptree(data_set, min_support, flag=False)  # 创建数据集的fptree
        # 利用递归条件子树的方式来挖掘频繁项集
        self.create_cond_fptree(headerTable, min_support, set(), freqItemSet, support_data)

        freq_item_N_set={}
        max_l = 0
        for i in freqItemSet:  # 统计得到项目最大的频繁项集的项数
            if len(i) > max_l: max_l = len(i)
        # 将频繁项集按照项目数统计
        freq_item_N_set = [dict() for _ in range(max_l)]
        for i in freqItemSet:
            freq_item_N_set [len(i) - 1].update({i:support_data[i]})
        return freq_item_N_set


if __name__ == "__main__":

    ##config
    # filename="药方.xls"
    # min_support=500#最小支持度
    # min_conf=0.9#最小置信度
    datas=[
        ['A','B','C','E','F','O'],
        ['A','C','G'],
        ['E','I'],
        ['A','C','D','E','G'],
        ['A','C','E','G','L'],
        ['E','J'],
        ['A','B','C','E','F','P'],
        ['A','C','D'],
        ['A','C','E','G','M'],
        ['A','C','E','G','N']
    ]
    fp = Fp_growth()
    rule_list = fp.rule_mining(datas, 2)
    print(rule_list)
