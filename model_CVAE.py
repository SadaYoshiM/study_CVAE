#Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

#Output
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import pprint

#変更可能なデータを一括ここの変数で管理する
conv1 = 64
conv2 = 32
latent_dim = 2
MAX_DATALENGTH = 100
MAX_TRAINLENGTH = 5
transform_list = ['w', '.', '+', 'g', '1', '2', '3', 'A']
symbol_size = len(transform_list)
DEVICE = 'cuda'

# Parameters
p_height = 9 #縦の長さ
p_width = 13 #横の長さ

px_Walls = 0.4 #壁の割合(任意指定定数)
px_Enemies = 0.2 #敵の割合(任意指定定数)
px_A2K = 0.3 #鍵までの距離比率(任意指定定数)
px_A2D = 0.4 #扉までの距離比率(任意指定定数)

p_rateWalls = 0.45+0.2*px_Walls #壁の割合
p_rateEnemies = 0.1+0.4*px_Enemies #敵の割合
p_distA2K = 0.1+0.8*px_A2K #鍵までの距離比率
p_distA2D = 0.1+0.8*px_A2D #扉までの距離比率

perRate = 1/(p_height*p_width)
original_dim = p_height * p_width
input_shape = (original_dim, )

log_path = "maps/train/log.txt"
learnp_path = "maps/train/learn.txt"
csv_path = "maps/train/csv/"

data_len = 5 #学習データ数
init_len = 16 #初期の生成データ長
newdata_len = 16 #新規生成データ長
label_len = 4 #ラベル長
bsize = 5 #バッチサイズ
num_epochs = 1000 #エポック数
losses = [] #損失
all_num = 5 #全データ数
gamma = 0.9 #学習率
F_const = [0.8, 0.8, 0.4, 0.4] #差分進化法の更新項の定数

#initial parameters
lp_wall = 0.5-0.25*(p_rateWalls-0.45)
lp_enemy = 0.5-0.125*(p_rateEnemies-0.1)
lp_A2K = 0.6-0.5*(p_distA2K-0.1)
lp_A2D = 0.6-0.5*(p_distA2D-0.1)

class Node():
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

def A_star2(stage, start, end):
    start_node = Node(None, (start[0], start[1]))
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, (end[0], end[1]))
    end_node.g = end_node.h = end_node.f = 0

    open_list = []
    closed_list = []
    
    open_list.append(start_node)

    while len(open_list) > 0:
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        open_list.pop(current_index)
        closed_list.append(current_node)

        #ゴールに到達していればTrueを返して終了
        if current_node == end_node:
            return abs(current_node.position[0]-start_node.position[0])+abs(current_node.position[1]-start_node.position[1])

        # ゴールに到達していなければ子ノードを生成
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])
            if node_position[0] <= 0 or node_position[0] >= p_height or node_position[1] <= 0 or node_position[1] >= p_width:
                continue
            if stage[node_position[0]][node_position[1]] != 0:
                new_node = Node(current_node, node_position)
                children.append(new_node)

        #f, g, h値を計算
        for child in children:
            if len([closed_child for closed_child in closed_list if closed_child == child]) > 0:
                continue
            child.g = current_node.g + 1
            child.h = abs(child.position[0] - end_node.position[0]) + abs(child.position[1] - end_node.position[1])

            child.f = child.g + child.h

            if len([open_node for open_node in open_list if child.position == open_node.position and child.g > open_node.g]) > 0:
                continue
            open_list.append(child)
            
    #dist = abs(current_node.position[0]-start_node.position[0])+abs(current_node.position[1]-start_node.position[1])
    return 0

def generateLabels(rpath, wpath, data, lpath):
    #Wall, Empty, Key, Exit door, Enemy1, Enemy2, Enemy3, Player, Path
    cntr = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    p_key = []
    p_goal = []
    p_avatar = []
    
    if rpath == "none" or rpath == "report":
        leveldata = data.copy()
        for i in range(p_height):
            for j in range(p_width):
                symbol = int(data[i][j])
                cntr[symbol] += 1
                if symbol == 2:
                    p_key = [i, j]
                if symbol == 3:
                    p_goal = [i, j]
                if symbol == 7:
                    p_avatar = [i, j]
    else:
        f = open(rpath)
        s = f.readlines()
        symbol = 0
        leveldata = np.zeros((p_height, p_width))

        for i in range(p_height):
            for j in range(p_width):
                if s[i][j] != '\n':
                    symbol = transform_list.index(s[i][j])
                    leveldata[i][j] = symbol
                else:
                    continue
                cntr[symbol] += 1
                if symbol == 2:
                    p_key = [i, j]
                if symbol == 3:
                    p_goal = [i, j]
                if symbol == 7:
                    p_avatar = [i, j]

        f.close()
            
    f = open(wpath, "w")
    # Rate of walls
    p_walls = p_height * p_width * p_rateWalls
    if cntr[0] >= p_walls-6 and cntr[0] <= p_walls+6:
        f.write("1")
    else:
        f.write("0")
    
    if rpath == "report":
        with open(lpath, "a") as pf:
            if cntr[0] < p_walls-6:
                pf.write("0l\n")
            elif cntr[0] > p_walls+6:
                pf.write("0h\n")
            else:
                pf.write("1\n")
        

    # Rate of enemies
    p_enemies = (p_height*p_width - cntr[0])*p_rateEnemies
    if (cntr[4] + cntr[5] + cntr[6]) >= p_enemies-5 and (cntr[4] + cntr[5] + cntr[6]) <= p_enemies+5:
        f.write("1")
    else:
        f.write("0")
    
    if rpath == "report":
        with open(lpath, "a") as pf:
            if (cntr[4] + cntr[5] + cntr[6]) < p_enemies-5:
                pf.write("0l\n")
            elif (cntr[4] + cntr[5] + cntr[6]) > p_enemies+5:
                pf.write("0h\n")
            else:
                pf.write("1\n")

    # A_star2(stage, start, end, dist)
    # return : distance
    # Rate of dictance between Player and Key
    distK = A_star2(leveldata, p_avatar, p_key)
    if distK != 0:
        #処理を実行
        p_A2K = (p_height+p_width-2) * p_distA2K
        if distK >= p_A2K-3 and distK <= p_A2K+3:
            f.write("1")
        else:
            f.write("0")
        
        if rpath == "report":
            with open(lpath, "a") as pf:
                if distK < p_A2K-3:
                    pf.write("0l\n")
                elif distK > p_A2K+3:
                    pf.write("0h\n")
                else:
                    pf.write("1\n")
        
    # Rate of dictance between Player and Door
    distD = A_star2(leveldata, p_avatar, p_goal)
    if distK != 0:
        #処理を実行
        p_A2D = (p_height+p_width-2) * p_distA2D
        if distD >= p_A2D-3 and distD <= p_A2D+3:
            f.write("1")
        else:
            f.write("0")
            
        if rpath == "report":
            with open(lpath, "a") as pf:
                if distD < p_A2D-3:
                    pf.write("0l\n")
                elif distD > p_A2D+3:
                    pf.write("0h\n")
                else:
                    pf.write("1\n")
    f.close()
    
def initMakeLabels(size):
    for n in range(int(size)):
        rpath = "maps/train/levels/zelda_lvl" + str(n) + ".txt"
        wpath = "maps/train/labels/zelda_lvl" + str(n) + ".txt"
        
        f = open(log_path, "a")
        f.write("initMakeLabels:" + str(n) + "\n")
        f.close()
        generateLabels(rpath, wpath, None, None)
        f = open(log_path, "a")
        f.write("completed.\n")
        f.close()

#既存データからゲームレベルを読み込む
def LoadLevels(size, pnum):
    read_data = np.zeros((int(size), symbol_size, p_height, p_width))
    for n in range(int(size)):
        path = "maps/train/levels/zelda_lvl" + str(n) + ".txt"
        f = open(path)
        s = f.readlines()
        symbol = 0

        for i in range(p_height):
            for j in range(p_width):
                if s[i][j] != '\n':
                    symbol = transform_list.index(s[i][j])
                else:
                    continue
                read_data[n][symbol][i][j] = 1.0
        f.close()
    return read_data

#既存データからラベルを読み込む
def LoadLabels(size, pnum):
    read_label = np.zeros((int(size), 4))
    for n in range(int(size)):
        path = "maps/train/labels/zelda_lvl" + str(n) + ".txt"
        f = open(path)
        s = f.readline()
        for i in range(4):
            if s[i] == "0":
                read_label[n][i] = 0
            else:
                read_label[n][i] = 1
        f.close()
    return read_label

def MakeLevels(data, size, num):
    if not os.path.isdir("maps/" + str(int(num))):
        os.mkdir("maps/" + str(int(num)))
    if not os.path.isdir("maps/" + str(int(num)) + "/levels"):
        os.mkdir("maps/" + str(int(num)) + "/levels")
        
    for n in range(int(size)):
        path = "maps/" + str(int(num)) + "/levels/zelda_lvl" + str(n) + ".txt"
        f = open(path, "w")

        for i in range(p_height):
            for j in range(p_width):
                f.write(transform_list[int(data[n][i][j])])
            f.write('\n')
        f.close()

def MakeLabels(data, size, num):
    if not os.path.isdir("maps/" + str(int(num))):
        os.mkdir("maps/" + str(int(num)))
    if not os.path.isdir("maps/" + str(int(num)) + "/labels"):
        os.mkdir("maps/" + str(int(num)) + "/labels")
    
    for n in range(int(size)):
        path = "maps/" + str(int(num)) + "/labels/zelda_lvl" + str(n) + ".txt"
        
        lpath = "maps/" + str(int(num)) + "/labels/learn" + str(n) + ".txt" 
        generateLabels("report", path, data[n], lpath)

def all_MakeLevels(data, fn, tn):
    for n in range(int(fn), int(tn)):
        path = "maps/train/levels/zelda_lvl" + str(n) + ".txt"
        f = open(path, "w")
        
        for i in range(p_height):
            for j in range(p_width):
                f.write(transform_list[int(data[n-int(fn)][i][j])])
            f.write('\n')
        f.close()
        
def all_MakeLabels(data, fn, tn):
    for n in range(int(fn), int(tn)):
        path = "maps/train/labels/zelda_lvl" + str(n) + ".txt"
        
        generateLabels("none", path, data[n-int(fn)], None)
        
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, label, transform=None):
        self.transform = transform
        self.data = data
        self.data_num = len(data)
        self.label = label
        
    def __len__(self):
        return self.data_num
    
    #torch.permute(transform(fixed_data), (1, 2, 0))
    def __getitem__(self, idx):
        if self.transform:
            out_data = torch.permute(self.transform(self.data), (1, 2, 0))[idx]
            #out_data = self.transform(self.data)[idx]
            out_label = self.label[idx]
        else:
            out_data = self.data[idx]
            out_label = self.label[idx]
            
        return out_data, out_label
    
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_lables):
        super(Encoder, self).__init__()
        self.num_labels = num_labels
        self.fc = nn.Linear(input_dim+self.num_labels, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, label):
        # ラベル
        label_onehot = torch.zeros(label.shape[0], self.num_labels).to(device)
        label_onehot.scatter_(1, label.unsqueeze(1), 1.0)
        x_cat = torch.cat((x, label_onehot), dim=-1)
        

        # ニューラルネットワーク
        h = torch.relu(self.fc(x_cat))
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)

        eps = torch.randn_like(torch.exp(log_var))
        z = mu + torch.exp(log_var / 2) * eps
        return mu, log_var, z
    
class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_labels):
        super(Decoder, self).__init__()
        self.num_labels = num_labels
        self.fc = nn.Linear(latent_dim+self.num_labels, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim, input_dim)

    def forward(self, z, label):
        # ラベル
        label_onehot = torch.zeros(label.shape[0], self.num_labels).to(device)
        label_onehot.scatter_(1, label.unsqueeze(1), 1.0)
        z_cat = torch.cat((z, label_onehot), dim=-1)
        h = torch.relu(self.fc(z_cat))
        output = torch.sigmoid(self.fc_output(h))
        return output

class CVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_labels):
        super(CVAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, num_labels)
        self.decoder = Decoder(input_dim, hidden_dim, latent_dim, num_labels)

    def forward(self, x, label):
        mu, log_var, z = self.encoder(x, label)
        x_decoded = self.decoder(z, label)
        return x_decoded, mu, log_var, z

def loss_function(label, predict, mu, log_var):
    reconstruction_loss = F.binary_cross_entropy(predict, label, reduction='sum')
    kl_loss = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    vae_loss = reconstruction_loss + kl_loss
    return vae_loss, reconstruction_loss, kl_loss

def generateLevel(gen_num, model, flag):
    if flag:
        labels = torch.tensor(np.ones(int(gen_num))).to(device)
    else:
        labels = torch.tensor(np.tile(np.arange(2), int(gen_num/2))).to(device)
    #labels = torch.tensor(np.tile(np.arange(2), int(gen_num/2))).to(device)
    labels = labels.to(torch.int64)
    model.eval()
    with torch.no_grad():
        z = torch.randn(gen_num, latent_dim).to(device)
        out = model.decoder(z, labels)
    out = out.view(-1, p_height, p_width)
    out = out.cpu().detach().numpy()
    return out

#Check playability
def checkPlayability(data):
    #Counter of symbls
    #Wall, Empty, Key, Exit door, Enemy1, Enemy2, Enemy3, Player
    cntr = [0, 0, 0, 0, 0, 0, 0, 0]
    p_key = [0, 0]
    p_goal = [0, 0]
    p_avatar = [0, 0]
    
    for i in range(p_height):
        for j in range(p_width):
            cntr[int(data[i][j])] += 1
            if data[i][j] == 2:
                p_key = [i, j]
            if data[i][j] == 3:
                p_goal = [i, j]
            if data[i][j] == 7:
                p_avatar = [i, j]
            
    #Numbre of enemies
    c_enms = cntr[4]+cntr[5]+cntr[6]
    
    #Check Playability
    if cntr[2]!=1 or cntr[3]!=1 or cntr[7]!=1:
        return False
    if c_enms >= (p_height*p_width-cntr[0])*6/10:
        return False
    if A_star2(data, p_avatar, p_key) <= 0:
        return False
    if A_star2(data, p_avatar, p_goal) <= 0:
        return False
    
    return True

update_list = [0.05, 0.05, 0.02, 0.02]
p_updated = [0, 0, 0, 0]
def fit_parameter(para, n, dn, index):
    p_updated[index] = para
    if dn <= 0:
        return para
    lpath = "maps/" + str(n) + "/labels/learn" + str(dn-1) + ".txt"
    with open(lpath, "r") as lf:
        s = lf.readlines()
        if s[index][0] == "0":
            update_list[index] *= gamma
            if s[index][1] == "l":
                p_updated[index] -= update_list[index]
            else:
                p_updated[index] += update_list[index]
    return p_updated[index]

def increaseMatrix(data):
    result = np.zeros((p_height, p_width))
    for i in range(1, p_height-1):
        for j in range(1, p_width-1):
            for k in range(-1, 2):
                for l in range(-1, 2):
                    result[i][j] += data[i+k][j+l]
    return result

#現在のエポックのラベルを読み込む
def e_LoadLabels(size, epoch):
    read_label = np.zeros((int(size), 4))
    for n in range(int(size)):
        path = "maps/" + str(epoch) + "/labels/zelda_lvl" + str(n) + ".txt"
        with open(path) as f:
            s = f.readline()
            for i in range(4):
                if s[i] == "0":
                    read_label[n][i] = 0
                else:
                    read_label[n][i] = 1
    return read_label


def random_nodup():
    r = set()
    res_list = [0, 0, 0]
    while len(r) < 3:
        r.add(np.random.randint(0, 9))
    for i in range(3):
        res_list[i] = r.pop()
    np.random.shuffle(res_list)
    return res_list

def differentialEvolution(size, epoch, lw, le, lk, ld, p_index):
    if size > 0:
        p_list = [lw, le, lk, ld]
        label = e_LoadLabels(size, epoch)
        label_count = [0, 0, 0, 0]
        cur_labelc = 0
        past_labelc = cur_labelc
        replace_index = p_index
        if epoch <= 1:
            past_labelc = 100
            
        for i in range(int(size)):
            for j in range(4):
                label_count[j] += label[i][j]
                if label[i][j]:
                    cur_labelc += 1
        if cur_labelc > past_labelc:
            for i in range(4):
                p_candidate[int(replace_index)][i] = new_y[i]
        
        past_labelc = cur_labelc
        
        max_i = label_count.index(max(label_count))
        
        diff_max = abs(p_list[max_i]-p_candidate[0][max_i])
        for i in range(8):
            diff = abs(p_list[max_i]-p_candidate[i+1][max_i])
            if diff_max < diff:
                diff_max = diff
                replace_index = i+1

        index_list = random_nodup()
        new_m = [0, 0, 0, 0]
        for i in range(4):
            new_m[i] = p_candidate[index_list[0]][i] + F_const[i] * (p_candidate[index_list[1]][i] - p_candidate[index_list[2]][i])
        for i in range(4):
            if np.random.rand() < 1 - label_count[i]/size:
                new_y[i] = new_m[i]
            else:
                new_y[i] = p_candidate[int(replace_index)][i]
            if i == max_i:
                new_y[i] = p_list[i]
    else:
        replace_index = np.random.randint(0, 9)
        for i in range(4):
            new_y[i] = p_candidate[replace_index][i]
    
    return new_y[0], new_y[1], new_y[2], new_y[3], replace_index

#学習と生成
initMakeLabels(5)
parameters = np.zeros((4, num_epochs))
p_candidate = [[lp_wall+np.random.uniform(-0.05, 0.05), lp_enemy+np.random.uniform(-0.05, 0.05), lp_A2K+np.random.uniform(-0.02, 0.02), lp_A2D+np.random.uniform(-0.02, 0.02)] for i in range(9)]

replace_index = 0
new_y = [0, 0, 0, 0]
cand_index = 0
for epoch in range(num_epochs):
    with open(log_path, "a") as lf:
        lf.write("Loop " + str(epoch) + "\n")
        lf.write("Now Data Length : " + str(all_num) + "\n")
        lf.write("\n##parameters##\n")
        lf.write("wall: " + str(lp_wall) + "\n")
        lf.write("enemies: " + str(lp_enemy) + "\n")
        lf.write("A2K: " + str(lp_A2K) + "\n")
        lf.write("A2D: " + str(lp_A2D) + "\n")
        lf.write("\n")
    
    for i in range(4):
        with open(csv_path+"parameters"+str(i)+".csv", "a", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([p_candidate[s][i] for s in range(len(p_candidate))])
    
    # モデル定義
    learning_rate = 1e-3
    num_labels = 2
    device = DEVICE
    model = CVAE(original_dim, conv2, latent_dim, num_labels).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    
    data_len = all_num
    bsize = int(data_len/5)
    
    # ゲームレベルを読み込む
    # return : numpy(datanum, symbol_size, p_height, p_width)
    data = LoadLevels(data_len, 0)
    
    # ラベルを読み込む
    # return : numpy(datanum, 4)
    label = LoadLabels(data_len, 0)
    
    # 前処理
    fixed_data = np.zeros((data_len, p_height, p_width))
    fixed_label = np.zeros((data_len))
    generated_data = np.zeros((label_len, init_len, p_height, p_width))
    gen_data = np.ones((init_len, p_height, p_width))
    for lb in range(label_len):
        if lb == 0:
            # 壁
            for i in range(data_len):
                fixed_data[i] = data[i][0]
                fixed_label[i] = label[i][0]
        elif lb == 1:
            # 敵
            for i in range(data_len):
                for n in range(4, 7):
                    fixed_data[i] += data[i][n]
                fixed_label[i] = label[i][1]
        elif lb == 2:
            # A2K
            for i in range(data_len):
                fixed_data[i] += data[i][1]
                fixed_data[i] *= lp_A2K
                if epoch > 4 and lp_A2K != parameters[2][epoch-1]:
                    data[i][2] = increaseMatrix(data[i][2])
                fixed_data[i] += data[i][2]
                if epoch > 4 and lp_A2K != parameters[2][epoch-1]:
                    data[i][7] = increaseMatrix(data[i][7])
                fixed_data[i] += data[i][7]
                fixed_label[i] = label[i][2]
        else:
            # A2D
            for i in range(data_len):
                fixed_data[i] += data[i][1]
                fixed_data[i] *= lp_A2D
                if epoch > 4 and lp_A2D != parameters[3][epoch-1]:
                    data[i][3] = increaseMatrix(data[i][3])
                fixed_data[i] += data[i][3]
                if epoch > 4 and lp_A2D != parameters[3][epoch-1]:
                    data[i][7] = increaseMatrix(data[i][7])
                fixed_data[i] += data[i][7]
                fixed_label[i] = label[i][3]
        
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = MyDataset(fixed_data, fixed_label, transform)
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=bsize, shuffle=False)

        #学習
        train_loss = 0
        for i, (x, labels) in enumerate(data_loader):
            x = x.to(device).view(-1, original_dim)
            x = x.to(torch.float32)
            labels = labels.to(device)
            labels = labels.to(torch.int64)
            x_recon, mu, log_var, z = model(x, labels)
            loss, recon_loss, kl_loss = loss_function(x, x_recon, mu, log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss)
            
        #生成
        newdata_len = init_len
        if data_len < 600:
            out = generateLevel(newdata_len, model, 0)
        else:
            out = generateLevel(newdata_len, model, 1)
            
        generated_data[lb] = out.copy()
        A1 = [0, 0]
        A2 = [0, 0]
        if lb == 0:#Wall
            walls = out.copy()
            for n in range(newdata_len):
                for i in range(p_height):
                    for j in range(p_width):
                        if out[n][i][j] >= lp_wall:
                            gen_data[n][i][j] = 0
        elif lb == 1:#Enemies
            for n in range(newdata_len):
                for i in range(p_height):
                    for j in range(p_width):
                        if out[n][i][j] >= lp_enemy:
                            gen_data[n][i][j] = np.random.randint(4, 7)
        elif lb == 2:#A2K
            for n in range(newdata_len):
                M1 = 0
                M2 = 0
                for i in range(p_height):
                    for j in range(p_width):
                        if i == 0 or i == p_height-1 or j == 0 or j == p_width-1:
                            continue
                        if M1 < out[n][i][j]:
                            M2 = M1
                            M1 = out[n][i][j]
                            A1 = [i, j]
                        elif M2 < out[n][i][j]:
                            M2 = out[n][i][j]
                
                for i in range(p_height):
                    for j in range(p_width):
                        if out[n][i][j] == M2:
                            gen_data[n][i][j] = 2
        else:#A2D
            for n in range(newdata_len):
                M1 = 0
                M2 = 0
                for i in range(p_height):
                    for j in range(p_width):
                        if i == 0 or i == p_height-1 or j == 0 or j == p_width-1:
                            continue
                        if M1 < out[n][i][j]:
                            M2 = M1
                            M1 = out[n][i][j]
                            A2 = [i, j]
                        elif M2 < out[n][i][j]:
                            M2 = out[n][i][j]
                
                for i in range(p_height):
                    for j in range(p_width):
                        if out[n][i][j] == M2:
                            gen_data[n][i][j] = 3
        
        for n in range(newdata_len):
            if A1[0] == A2[0] and A1[1] == A2[1]:
                gen_data[n][A1[0]][A1[1]] = 7
            else:
                if np.random.random() > 0.5:
                    gen_data[n][A1[0]][A1[1]] = 7
                else:
                    gen_data[n][A2[0]][A2[1]] = 7
    
    #四方を壁に変更する処理
    for n in range(newdata_len):
        for i in range(p_height):
            gen_data[n][i][0] = 0
            gen_data[n][i][p_width-1] = 0
        for j in range(p_width):
            gen_data[n][0][j] = 0
            gen_data[n][p_height-1][j] = 0
    
    # プレイアブルチェック
    new_data = []
    for n in range(newdata_len):
        #check
        if checkPlayability(gen_data[n]):
            new_data.append(gen_data[n])
        else:
            newdata_len -= 1
    
    #保存＆ラベル付与
    data_len = newdata_len
    MakeLevels(new_data, data_len, epoch+1)
    all_MakeLevels(new_data, all_num, all_num+data_len)
    MakeLabels(new_data, data_len, epoch+1)
    all_MakeLabels(new_data, all_num, all_num+data_len)
    all_num += data_len
    
    
    #ここを変更します！！
    lp_wall, lp_enemy, lp_A2K, lp_A2D, cand_index = differentialEvolution(data_len, epoch+1, lp_wall, lp_enemy, lp_A2K, lp_A2D, cand_index)
    
    parameters[0][epoch] = lp_wall
    parameters[1][epoch] = lp_enemy
    parameters[2][epoch] = lp_A2K
    parameters[3][epoch] = lp_A2D


#データ書き出し
alls = np.zeros((4))
all4_num = []
all_count = 0
count = 0
cnt3 = []

for n in range(1,num_epochs+1):
    for lvl in range(16):
        label_path = "maps/" + str(n) + "/labels/zelda_lvl" + str(lvl) + ".txt"
        if not os.path.isfile(label_path):
            continue
        all_count += 1
        f = open(label_path, "r")
        s = f.readlines()
        count = 0
        for i in range(4):
            if s[0][i] == '1':
                alls[i] += 1
                count += 1
            if count >= 4:
                all4_num.append([n, lvl])
        if count == 3:
            cnt3.append([n, lvl])
        f.close()

with open("maps/result_4labels.txt", "w") as f:
    for s in range(4):
        f.write(str(alls[s]) + "\n")
    f.write("All count is " + str(all_count) + "\n")

with open("maps/result_3clear.txt", "w") as f:
    for s in range(len(cnt3)):
        f.write(str(cnt3[s]) + "\n")
        
with open("maps/result_clear.txt", "w") as f:
    for s in range(len(all4_num)):
        f.write(str(all4_num[s])+"\n")


x = np.arange(num_epochs)
fig = plt.figure(figsize=(24,16))

ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

ax1.plot(x, parameters[0])
ax2.plot(x, parameters[1])
ax3.plot(x, parameters[2])
ax4.plot(x, parameters[3])

fig.savefig("maps/trans_para.png")
plt.close(fig)

#ラベル変化
transition_label = np.zeros((4, all_num))
counter = 0

for n in range(1, num_epochs+1):
    for lvl in range(16):
        label_path = "maps/" + str(n) + "/labels/learn" + str(lvl) + ".txt"
        if not os.path.isfile(label_path):
            continue
        with open(label_path) as f:
            s = f.readlines()
            for i in range(4):
                if s[i][0] == '1':
                    transition_label[i][counter] = 0
                else:
                    if s[i][1] == 'l':
                        transition_label[i][counter] = -0.5
                    else:
                        transition_label[i][counter] = 0.5
        counter += 1
                        
x = np.arange(all_num)
fig = plt.figure(figsize=(64, 48))

ax1 = fig.add_subplot(4, 1, 1)
ax1.set_ylim(-1, 1)
ax2 = fig.add_subplot(4, 1, 2)
ax2.set_ylim(-1, 1)
ax3 = fig.add_subplot(4, 1, 3)
ax3.set_ylim(-1, 1)
ax4 = fig.add_subplot(4, 1, 4)
ax4.set_ylim(-1, 1)

ax1.plot(x, transition_label[0])
ax2.plot(x, transition_label[1])
ax3.plot(x, transition_label[2])
ax4.plot(x, transition_label[3])

fig.savefig("maps/trans_label.png")
plt.close(fig)