import cv2
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
from MIN2_ignore_sunspots import MIN2_ignore_sunspots as MIN2_ver1
class seeing_debug_tools:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    def where_diff_grad_smale(img:np.ndarray,
                                min2_edge:tuple,#(x,y)
                                sample_lib:list, #(startpoint(x,y),endpoint(x,y))
                                cir_stat:tuple,#(cx,cy),r
                                gradindx:float|int=None,
                                diffindx:float|int=None,
                                error:str="None",
                                title:str="None"):
            """主にdebug用です。sampleが画像上のどこなのかを見せてくれます。"""
            #grad,diffの位置を確認する際に描画の問題なのか実際の場所が違うのかを確認するためにindxを受け取る
            print("sample lib",sample_lib)
            print("gradindx(fis)",gradindx)
            print("diffindx(sed)",diffindx)
            (cx,cy),r=cir_stat
            fig, ax = plt.subplots()
            ax.imshow(img, cmap='magma')
            sample_lib=[sample_lib[0][0],sample_lib[1][0]],[sample_lib[0][1],sample_lib[1][1]]
            ax.plot( sample_lib[0],sample_lib[1], color='#09fb7e', label='sample', alpha=0.8, linewidth=2)
            ax.scatter( min2_edge[0], min2_edge[1], color='cyan', label='MIN edge', s=15)
            ax.add_patch(patches.Circle((cx, cy), r, fill=False, edgecolor='yellow', linewidth=2))
            if gradindx is not None:
                if sample_lib[1][0]==sample_lib[1][1]:#水平
                    print("horizontal") if debug else None
                    gradxy=sample_lib[0][0]+gradindx+0.5,sample_lib[1][1]
                else:#垂直
                    print("vertical") if debug else None
                    gradxy=sample_lib[0][0],sample_lib[1][0]+gradindx+0.5
                ax.scatter( gradxy[0], gradxy[1], color='red', label='Gradient max-first', s=25)
            if diffindx is not None:
                if sample_lib[1][1]==sample_lib[1][0]:#水平
                    print("horizontal") if debug else None
                    diffxy=sample_lib[0][0]+diffindx+0.5,sample_lib[1][1]
                else:#垂直
                    print("vertical") if debug else None
                    diffxy=sample_lib[0][0],sample_lib[1][0]+diffindx+0.5
                ax.scatter( diffxy[0], diffxy[1], color='blue', label='Difference max-second', s=25)
            fig.text(0.99, 0.01, f'error reason: {error}', ha='right', va='bottom', fontsize=15, color='gray')
            fig.suptitle(title)
            ax.legend()
            plt.show(block=True)
            return None
    def show_samples(limb_wigth,sample,place,e,gaps):
                    #縁サンプルの折れ線
                    fig, ax = plt.subplots()
                    # タイトル
                    fig.suptitle(f"{place}_{gaps}")
                    fig.text(0.5, 0.92, f"error reason: {e}", ha='center')
                    # 第2軸（右）
                    ax.plot(sample, color="green", label="sample_ax1",linewidth=0.7,alpha=0.7)
                    # 第1軸（左）
                    ax2 = ax.twinx()
                    ax2.plot(np.gradient(sample), label="grad(sample)_ax2",linewidth=0.7)
                    ax2.plot(np.diff(np.gradient(sample)), label="diff(grad(sample))_ax2",linewidth=0.7)
                    ax2.scatter(np.argmax(np.gradient(sample)), np.gradient(sample)[np.argmax(np.gradient(sample))], color="red", label="Gradient max(ax2)")
                    ax2.scatter(np.argmax(np.diff(np.gradient(sample))), np.diff(np.gradient(sample))[np.argmax(np.diff(np.gradient(sample)))], color="blue", label="Difference max(ax2)")
                    #min2の縁をプロット
                    ax2.axvline(x=limb_wigth, color="gray", label="min2 edge", linestyle="--")
                    # 凡例をまとめる
                    lines1, labels1 = ax.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
                    plt.show(block=True)
def seeing_one_frame(readed_img,cir_stat,limb_wigth=24,allp_num=1360,show=False,debug=False):
    """
    reimg:cv2で読み込んだ画像を渡してください
    cir_stat:MIN2_ignore_sunspotsの返り値を渡してください。
    limb_wigth: limbの解析幅
    show:解析を図示するか
    debug:途中経過を表示するか
    """
    realrlst=[]
    min2rlst=[]
    x,y=[],[] if show else (None,None)
    center,r=cir_stat
    cx,cy=center
    if allp_num%4!=0:
        raise("allp_numは4の倍数にしてください")
    for i in range(int(-(allp_num/8)),int(allp_num/8)):
        min2r=np.sqrt(r**2-i**2)#円の中心をとおる水平、垂直の線から目標円周上の点までの距離(近似円)
        min2rlst+=[float(Decimal(str(min2r)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))]#小数点第一位で四捨五入
        min2r=Decimal(str(min2r)).quantize(Decimal('1'), rounding=ROUND_HALF_UP)#整数に四捨五入、roundは銀行丸目なので注意
        min2r=float(min2r)#decimalはfloatと計算できないのでfloatに変換
        #L
        samples=readed_img[int(cy+i),int(cx-min2r-limb_wigth):int(cx-min2r+limb_wigth)]
        realindx=np.argmax(np.diff(np.gradient(samples)))+0.5#diffで要素が減る分
        realrlst+=[min2r+realindx-limb_wigth]
        if show:
            x+=[cx-min2r-limb_wigth+realindx]
            y+=[cy+i]
        #R
        samples=readed_img[int(cy+i),int(cx+min2r-limb_wigth):int(cx+min2r+limb_wigth)]
        realindx=np.argmax(np.diff(np.gradient(samples)))+0.5#diffで要素が減る分
        realrlst+=[min2r+realindx-limb_wigth]
        if show:
            x+=[cx+min2r-limb_wigth+realindx]
            y+=[cy+i]
        #T
        samples=readed_img[int(cy-min2r-limb_wigth):int(cy-min2r+limb_wigth),int(cx+i)]
        realindx=np.argmax(np.diff(np.gradient(samples)))+0.5#diffで要素が減る分
        realrlst+=[min2r+realindx-limb_wigth]
        if show:
            x+=[cx+i]
            y+=[cy-min2r-limb_wigth+realindx]
        #B
        samples=readed_img[int(cy+min2r-limb_wigth):int(cy+min2r+limb_wigth),int(cx+i)]
        realindx=np.argmax(np.diff(np.gradient(samples)))+0.5#diffで要素が減る分
        realrlst+=[min2r+realindx-limb_wigth]
        if show:
            x+=[cx+i]
            y+=[cy+min2r-limb_wigth+realindx]
    print(f"min2r\nmin:{np.min(min2rlst)},max:{np.max(min2rlst)},mean:{np.mean(min2rlst)},std:{np.std(min2rlst)}") if debug else None
    std=np.std(np.array(realrlst)-np.repeat(np.array(min2rlst),4))
    return std if not show else (std,x,y)

def seeing_show(readed_img,x,y,cir):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    center,r=cir
    figure, ax = plt.subplots()
    ax.imshow(readed_img, cmap='magma')
    ax.scatter(x, y, color='cyan', label='Detected edge points', s=10)
    circle = patches.Circle(center, r, fill=False, edgecolor='yellow', linewidth=2)
    ax.add_patch(circle)
    plt.show()

if __name__ == "__main__":
    from tkinter.filedialog import askopenfilename as ask
    import os
    import pandas as pd
    debug=True
    show=True
    filename=ask(title="画像を選択してください", filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.tiff")])
    if debug:    
        from time import time
        st=time()
    readed_img=cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    cir_stat=MIN2_ver1(((readed_img >> 8).astype("uint8")),n=10,light_threshold=50,limb_wigth=24,show=show,debug=debug)
    if show:
        std,x,y=seeing_one_frame(readed_img,cir_stat,limb_wigth=24,allp_num=1360,show=True,debug=debug)
        print(f"std:{std}")
        if debug:
            print(pd.DataFrame({"cx":cir_stat[0][0],"cy":cir_stat[0][1],"r":cir_stat[1]},index=[0]))    
            Df = pd.DataFrame({"x":pd.Series(x).agg(["min","max","mean","std"]),"y":pd.Series(y).agg(["min","max","mean","std"])})
            print(Df)
        print(f"seeing_one_frameの解析時間:{time()-st}秒") if debug else None
        seeing_show(readed_img,x,y,cir_stat)
    else:
        print(seeing_one_frame(readed_img,cir_stat,limb_wigth=24,allp_num=1360,show=False,debug=debug))
        print(f"seeing_one_frameの解析時間:{time()-st}秒") if debug else None
    