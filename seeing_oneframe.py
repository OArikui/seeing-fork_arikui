import cv2
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
from MIN2_ignore_sunspots import MIN2_ignore_sunspots as MIN2_ver1
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
    cx,cy,r=cir_stat
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
    cx,cy,r=cir
    figure, ax = plt.subplots()
    ax.imshow(readed_img, cmap='magma')
    ax.scatter(x, y, color='cyan', label='Detected edge points', s=10)
    circle = patches.Circle((cx, cy), r, fill=False, edgecolor='yellow', linewidth=2)
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
            print(pd.DataFrame({"cx":cir_stat[1],"cy":cir_stat[2],"r":cir_stat[0]},index=[0]))
            Df = pd.DataFrame({"x":pd.Series(x).agg(["min","max","mean","std"]),"y":pd.Series(y).agg(["min","max","mean","std"])})
            print(Df)
        print(f"seeing_one_frameの解析時間:{time()-st}秒") if debug else None
        seeing_show(readed_img,x,y,cir_stat)
    else:
        print(seeing_one_frame(readed_img,cir_stat,limb_wigth=24,allp_num=1360,show=False,debug=debug))
        print(f"seeing_one_frameの解析時間:{time()-st}秒") if debug else None
    