
import cv2
import numpy as np
import matplotlib.pyplot as plt
"""
最小二乗法による円の検出を行う関数を作成します。
一度検出した近似円の内側にある点のうち、近似円の外側にある点だけから近似した円からlimbwigth*(3/2)の
範囲にないもんは黒点とみなします。
"""

def cut_and_sampling(sun_threshold):#imgの画像をnで分割、太陽の縁の点をsampling
    #画像を分割して実際の縁の点を収集
    spots=[]#実際の縁の点を格納するための配列
    for line_xy in ("x_line","y_line"):#x_lineは横線、y_lineは縦線
        for i in range(1,divnum):
            place = height * i // divnum      if line_xy == "x_line" else width * i // divnum#分割線の位置を計算
            line = img[place,:].astype(float) if line_xy == "x_line" else img[:, place].astype(float)#分割線に沿った画素値を取得
            if np.max(line) <= sun_threshold:#太陽像上を通るか
                continue
            grad_t = np.diff(line)#一回微分
            max_idx = int(np.argmax(grad_t)) # 最大値のインデックス
            min_idx = int(np.argmin(grad_t)) # 最小値のインデックス
            if line_xy=="x_line":
                spots.append([max_idx, place])
                spots.append([min_idx, place])
            elif line_xy=="y_line":
                spots.append([place, max_idx])
                spots.append([place, min_idx])
    return spots#縁の点の座標を返す

def fit_circle(spots):#縁の点のサンプルを受け取って円のstatusを返す。
    if len(spots)<3:
        print("点が3点未満のため、円を作成できません。")
        raise(f"点不足{show_circle(spots,False)}")
    x,y=np.array([s[0] for s in spots],dtype=float),np.array([s[1] for s in spots],dtype=float)
    mat_A = np.c_[x, y, np.ones(len(x))]
    vec_B = -(x**2 + y**2)
    res, _, _, _ = np.linalg.lstsq(mat_A, vec_B, rcond=None)
    A, B, C = res
    cx = -A / 2
    cy = -B / 2
    R = np.sqrt(cx**2 + cy**2 - C)
    return [cx,cy,R]

def show_circle(spots=[],cir_stat=False):#円のstatusを受け取って、画像に円と縁の点を描画して表示する。
    fig, ax = plt.subplots()#figとaxの作成
    ax.imshow(img, cmap="magma")#画像をグレースケールで表示
    if cir_stat != False:#cir_statがFalseでないなら、円を描画
        cx,cy,R=cir_stat[0],cir_stat[1],cir_stat[2]
        circle = plt.Circle((cx, cy), R, fill=False, color='orange', linewidth=2)#結果の円を描画
        ax.add_patch(circle)###
    if len(spots) > 0:
        x, y = zip(*spots)
        ax.scatter(x, y, color='red', label='Edges', s=50)
    # 座標ラベルを表示
    for xi, yi in zip(x, y):
        ax.text(xi, yi, f"({xi:.0f}, {yi:.0f})",
                color="#8917fd", fontsize=8,
                ha="left", va="bottom")

    #画像の分割線を描画
    lines=[]
    for xy in ["x","y"]:#各分割線のlistを作成
        lines.append([])
        for nn in range(divnum) :
            ap=width/divnum if xy =="x" else height/divnum
            lines[-1].append(ap*(nn+1))
    for li in lines[0]:#x方向の分割線を描画
        ax.axvline(int(li), color='white', linestyle='--', alpha=0.3)
    for li in lines[1]:#y方向の分割線を描画
        ax.axhline(int(li), color='white', linestyle='--', alpha=0.3)

    # nの値を左上に固定表示
    ax.text(.05,.9, f"n={divnum}",
            color="cyan", fontsize=10,
            transform=ax.transAxes)
    ax.legend()###
    ax.axis('equal')###
    plt.show()#windowで表示

def MIN2_ignore_sunspots(readed_img:np.ndarray,n=10,light_threshold=50,filter_times=3,limb_wigth=24,show=False,debug=False):
    """
    img:読み込んだ画像を渡してください
    n:画像格子の分割数
    light_threshold:太陽の明るさの基準です。

    """
    #===基本的な変数をglobalで宣言===
    global divnum#分割数、引数ではnとして受け取っている。
    divnum=n
    global img#読み込んだ画像
    img=readed_img
    global height, width#画像の高さと幅
    height, width = img.shape
    
    #円の情報[cx, cy, R]
    spots=cut_and_sampling(light_threshold)#spots=[[x1,y1],[x2,y2],...]の形式で、縁の点の座標を格納したlist
    cx,cy,r=fit_circle(spots)#一回目の円情報
    print(f"first circle{show_circle(spots,(cx,cy,r))}") if debug and show else None#一回目の円を表示debug用
    for i in  range(filter_times):#何度か繰り返すことでふくすうの黒点にも対応
        #===MIN2の外側の点を抽出===
        outside_spots=[]
        for i in range(len(spots)):
            if int(((spots[i][0]-cx)**2+(spots[i][1]-cy)**2)**(1/2))>r:
                outside_spots.append(spots[i])
        if len(outside_spots)>2:
            #外側の点が３つ以上ないと以下の解析はできないが、 そもそもそのような場合は、黒点の影響は受けていない
            cxo,cyo,ro=fit_circle(np.array(outside_spots, dtype=float))#外側の点だけで円を作成
            print(f"only outside circle{show_circle(outside_spots,(cxo,cyo,ro))}") if debug and show else None#外側の点だけで作成した円を表示
            
            #===外側の点の円からlimb_wigthの範囲にない内側の点を抽出===
            sunspot=False
            print (f"外側の点の数:{len(outside_spots)},全体の点の数:{len(spots)}") if debug else None
            for i in range(len(spots)):
                x=spots[i][0]
                y=spots[i][1]
                if not spots[i] in outside_spots:#内側の点だけ
                    if (x-cxo)**2 > (y-cyo)**2:#円のRLTBのうちRLなら、
                        min2far = np.sqrt(ro**2-(y-cyo)**2)
                        print(f"x,y:{x,y} min2far:{min2far},y-cyo:{np.abs(cyo-y)}") if debug else None
                        if min2far-np.abs(cxo-x) > limb_wigth:
                            sunspot=True
                            spots.pop(i)
                    else:#円のRLTBのうちTBなら
                        min2far = np.sqrt(ro**2-(x-cxo)**2)
                        print(f"x,y:{x,y} min2far:{min2far},x-cxo:{np.abs(cxo-x)}") if debug else None
                        if min2far-np.abs(cyo-y) > limb_wigth:
                            sunspot=True
                            spots.pop(i)
            if sunspot:    
                #黒点とみなされない点だけで円を作成
                cx,cy,r=fit_circle(np.array(spots, dtype=float))
            else:
                break
        else:
            break 
        if show:
            show_circle(spots,(cx,cy,r))
    return (cx,cy),r

if __name__== "__main__":
    from tkinter.filedialog import askopenfilename
    picpath=askopenfilename(title="画像を選択してください", filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.tiff")])
    from time import time
    start = time()
    print("cx,cy,r=",MIN2_ignore_sunspots(cv2.imread(picpath,0),filter_times=2,show=False,debug=True))
    print(f"処理時間:{time()-start}秒")