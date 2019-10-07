class Astar:
    def __init__(self):
        self.tar_dist = []
        self.city_name = []
        self.city_index = {}
        self.shortest_num = []
        self.shortest_path = []
        self.openlist = []

        """
        a ~ z
        0 ~ 19
        """
        self.graph = [[0 for i in range(20)] for i in range(20)]
        self.graph[0][19] = self.graph[19][0] = 75
        self.graph[0][16] = self.graph[16][0] = 118
        self.graph[0][15] = self.graph[15][0] = 140
        self.graph[1][17] = self.graph[17][1] = 85
        self.graph[1][5] = self.graph[5][1] = 211
        self.graph[1][6] = self.graph[6][1] = 90
        self.graph[1][13] = self.graph[13][1] = 101
        self.graph[2][13] = self.graph[13][2] = 138
        self.graph[2][14] = self.graph[14][2] = 146
        self.graph[2][3] = self.graph[3][2] = 120
        self.graph[3][10] = self.graph[10][3] = 75
        self.graph[4][7] = self.graph[7][4] = 86
        self.graph[5][15] = self.graph[15][5] = 99
        self.graph[7][17] = self.graph[17][7] = 98
        self.graph[8][11] = self.graph[11][8] = 87
        self.graph[8][18] = self.graph[18][8] = 92
        self.graph[9][10] = self.graph[10][9] = 70
        self.graph[9][16] = self.graph[16][9] = 111
        self.graph[12][15] = self.graph[15][12] = 151
        self.graph[12][19] = self.graph[19][12] = 71
        self.graph[13][14] = self.graph[14][13] = 97
        self.graph[14][15] = self.graph[15][14] = 80
        self.graph[17][18] = self.graph[18][17] = 142
        
        self.tar_dist = [366,0,160,242,161,178,77,151,226,244,241,234,380,98,193,253,329,80,199,374]
        self.city_name = ['Arad','Bucharest','Craiova','Dibreta','Eforie',
            'Fafarsas','Giurgiu','Hirsova','Iasi','Lugoj',
            'Mehadia','Neamt','Oradea','Pitesti','Rimnicu_Vilcea',
            'Sibiu','Timisoara','Urziceni','Vaslui','Zerind']
        self.city_index = {'Arad':0,'Bucharest':1,'Craiova':2,'Dibreta':3,'Eforie':4,
            'Fafarsas':5,'Giurgiu':6,'Hirsova':7,'Iasi':8,'Lugoj':9,
            'Mehadia':10,'Neamt':11,'Oradea':12,'Pitesti':13,'Rimnicu_Vilcea':14,
            'Sibiu':15,'Timisoara':16,'Urziceni':17,'Vaslui':18,'Zerind':19}
        self.shortest_num = [999 for i in range(20)]
        self.shortest_path = [[] for i in range(20)] # 每一个元素是一个列表，记录从开头到这里的路径
        self.openlist = self.city_name.copy()
        self.closelist = []
        self.mincost = 999
        self.min_col = 0
        self.path = ["Arad"]

    # def set_list(self, current, openlist):
    #     self.openlist = openlist
    #     for i in range(20):
    #         if self.graph[self.city_index[current]][i] != 0:
    #             if self.city_name[i] in self.closelist:
    #                 pass
    #             else:
    #                 self.openlist.append(self.city_name[i])

    def astar(self, start, target, openlist):
        for i in range(20):
            if self.graph[self.city_index[start]][i]>0:
                self.shortest_num[i] = self.graph[self.city_index[start]][i] # shortest_num存的是"直接"到这一点的距离
                self.shortest_path[i].append(start)
                self.shortest_path[i].append(self.city_name[i])

        while len(openlist):
            current = self.find_min(openlist)

            if current == -1:
                break
            openlist.remove(self.city_name[current])
            for i in range(20):
                if self.graph[current][i] > 0 and self.graph[current][i] + self.shortest_num[current] < self.shortest_num[i]:
                    self.shortest_num[i] = self.graph[current][i] + self.shortest_num[current]
                    self.shortest_path[i] = self.shortest_path[current].copy()
                    self.shortest_path[i].append(self.city_name[i])
                    if (self.city_name[i] == target):
                        openlist = []
                        break

    def find_min(self, openlist):
        minc = 999
        current = -1
        for city in openlist:
            if self.shortest_num[self.city_index[city]] + self.tar_dist[self.city_index[city]] < minc:
                minc = self.shortest_num[self.city_index[city]] + self.tar_dist[self.city_index[city]]
                current = self.city_index[city]
        return current

    def find(self, start, target):
        self.start = self.city_index[start]
        self.shortest_num[self.start] = 0
        self.openlist.pop(self.start)
        self.astar(start, target, self.openlist)
        print("The Path is: ")
        for i in range(len(self.shortest_path[self.city_index[target]])-1):
            print(self.shortest_path[self.city_index[target]][i] + ' -> ',end='')
        print(self.shortest_path[self.city_index[target]][len(self.shortest_path[self.city_index[target]])-1])
        print("The cost is: " + str(self.shortest_num[self.city_index[target]]))


test = Astar()
test.find('Arad', 'Bucharest')
