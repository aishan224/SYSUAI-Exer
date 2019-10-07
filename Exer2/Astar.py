"""
a ~ z
0 ~ 19
"""
graph = [[0 for i in range(20)] for i in range(20)]
graph[0][19] = graph[19][0] = 75
graph[0][16] = graph[16][0] = 118
graph[0][15] = graph[15][0] = 140
graph[1][17] = graph[17][1] = 85
graph[1][5] = graph[5][1] = 211
graph[1][6] = graph[6][1] = 90
graph[1][13] = graph[13][1] = 101
graph[2][13] = graph[13][2] = 138
graph[2][14] = graph[14][2] = 146
graph[2][3] = graph[3][2] = 120
graph[3][10] = graph[10][3] = 75
graph[4][7] = graph[7][4] = 86
graph[5][15] = graph[15][5] = 99
graph[7][17] = graph[17][7] = 98
graph[8][11] = graph[11][8] = 87
graph[8][18] = graph[18][8] = 92
graph[9][10] = graph[10][9] = 70
graph[9][16] = graph[16][9] = 111
graph[12][15] = graph[15][12] = 151
graph[12][19] = graph[19][12] = 71
graph[13][14] = graph[14][13] = 97
graph[14][15] = graph[15][14] = 80
graph[17][18] = graph[18][17] = 142

h = [366,0,160,242,161,178,77,151,226,244,241,234,380,98,193,253,329,80,199,374]
namelist = ['Arad','Bucharest','Craiova','Dibreta','Eforie',
     'Fafarsas','Giurgiu','Hirsova','Iasi','Lugoj',
     'Mehadia','Neamt','Oradea','Pitesti','Rimnicu_Vilcea',
     'Sibiu','Timisoara','Urziceni','Vaslui','Zerind']

cost = 0
row = 0
minc = 999
path = []
min_col = 0

def Astar(src, target):
    global graph
    global minc
    global min_col
    global path
    global row
    global cost
    path.append(src)
    while True:
        if namelist[min_col] == 'Bucharest':
            break
        for i in range(20):
            if graph[row][i]>0:
                temp = graph[row][i] + h[i]
                if temp < minc:
                    minc = temp
                    min_col = i
        cost += graph[row][min_col]
        row = min_col
        path.append(namelist[min_col])
        

def printpath():
    global path
    for i in range(len(path)-1):
        print(path[i]+' -> ',end='')
    print(path[len(path)-1])


def main():
    global cost
    print("The shortest path is: ")
    Astar('Arad', 'Bucharest')
    printpath()
    print("The minimum cost is: " + str(cost))

if __name__ == '__main__':
    main()
