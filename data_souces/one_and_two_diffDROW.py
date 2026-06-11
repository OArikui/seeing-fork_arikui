import cv2
import numpy as np
import matplotlib.pyplot as plt
from KANSUU import list_folders
from glob import glob
import os
from MIN2 import MIN2

"""
２回微分をふちの定義とする根拠のために、太陽画像上に1,2回微分の縁を両方描画するコードです。
"""

print(os.getcwd())
sun_image=glob(f"{os.getcwd()}\\data_souces\\"+"*.tiff")[0]
def sengiri_fissec_edges(    
    file
    ,gap=170
    ,limb_wigth = 24 
    ,look=False
    ):
    """
    一回微分の最大の点と、それよりも中心から外側の範囲での2回微分の最大の点の両方の位置を返します。
    """
    twox,twoy=[],[]
    onex,oney=[],[]
    if file.endswith(".tiff" or ".tif"):
        lst = np.zeros(gap*8)
        (xc, yc), r = MIN2(file)
        img= cv2.imread(file,cv2.IMREAD_UNCHANGED)
        for gaps in range(-gap , gap):
        
            #plt.clf()
            #太陽の縁を探索
            gaps_r =int(np.sqrt(np.abs(r**2 - gaps**2)))
            gaps_r = int(gaps_r)
            gaps = int(gaps)
            #  memo img[top : bottom, left : right]
            
            pl=["L","R","T","B"]
            xyof=[
                (gaps,gaps+1)
                ,[(-gaps_r-limb_wigth,-gaps_r+limb_wigth),(gaps_r-limb_wigth,gaps_r+limb_wigth)]
            ]
            #縦倒し,逆順
            pldc={
                "L":(True,0),
                "R":(True,1),
                "T":(False,0),
                "B":(False,1)
                }
            for place in pl:
                
                pld = pldc[place]
                "↑参照しておくよん"
                if pld[0]:
                    y_of,x_of = xyof[0],xyof[1][pld[1]] 
                elif not pld[0]:
                    y_of,x_of = xyof[1][pld[1]],xyof[0]
                
                splrang=( int(yc+y_of[0]), int(yc+y_of[1]), int(xc+x_of[0]), int(xc+x_of[1]) )
                
                #samplerangeが画像の範囲外になっていないか確認
                if splrang[0] < 0 or splrang[1] > img.shape[0] or splrang[2] < 0 or splrang[3] > img.shape[1]:
                    print("image shape:", img.shape)
                    print("sample range:", splrang)
                    optwidth=limb_wigth+min([img.shape[0]-(yc+y_of[1]),yc+y_of[0], img.shape[1]-(xc+x_of[1]), xc+x_of[0]])
                    print("suggested Optimal limb_wigth at this point:", int(optwidth))
                    raise IndexError(f"Sample range is out of image bounds for place: {place}, gap: {gaps}\nRead the termination message for details.")
                
                sample=img[splrang[0]:splrang[1], splrang[2]:splrang[3]]
                sample=np.array(np.ravel(sample))
                if bool(pld[1]):
                    sample=np.flip(sample)
                fd_sample = np.abs(np.gradient(sample))
                fMaxd_sample = np.amax(fd_sample)
                findex_Max_d_sample = np.where(fd_sample == fMaxd_sample)[0][0]
                if bool(pld[1]):
                    fd_sample=np.flip(fd_sample)
                ftruthindex_Max_d_sample = np.where(fd_sample == fMaxd_sample)[0][0]
                d_sample = np.abs(np.gradient(np.gradient(sample)))
                rowd_sample = d_sample
                d_sample=d_sample[:findex_Max_d_sample]
                Maxd_sample=np.max(d_sample)
                if bool(pld[1]):
                    rowd_sample=np.flip(rowd_sample)
                index_Max_d_sample = np.where(rowd_sample == Maxd_sample)[0][0]
                if look:
                    fig, ax = plt.subplots()
                    ax.set_title(f"{file.split('\\')[-1]}_{place}_{gaps}")
                    # sampleをプロット
                    ax.plot(sample, color="green", label="sample")
                    ax2 = ax.twinx()
                    ax.plot(fd_sample, label="fd_sample")
                    ax.plot(rowd_sample, label="d_sample")
                    ax.scatter(findex_Max_d_sample, fMaxd_sample, color="red", label="fd max")
                    ax.scatter(index_Max_d_sample, Maxd_sample, color="blue", label="d max")
                    ax.legend(loc="upper right")
                
                plt.show()
                if not bool(pld[1]):
                    index_Max_d_sample = 2*limb_wigth-index_Max_d_sample
                    ftruthindex_Max_d_sample = 2*limb_wigth-ftruthindex_Max_d_sample
                real_r = gaps_r - limb_wigth +index_Max_d_sample #limbまでの距離
                freal_r = gaps_r - limb_wigth +ftruthindex_Max_d_sample#limbまでの距離
                if place == "L":
                    twox.append(xc-real_r)
                    twoy.append(yc+gaps)
                    onex.append(xc+-freal_r)
                    oney.append(yc+gaps)
                elif place == "R":
                    twox.append(xc+real_r)
                    twoy.append(yc+gaps)
                    onex.append(xc+freal_r)
                    oney.append(yc+gaps)
                elif place == "T":
                    twox.append(xc+gaps)
                    twoy.append(yc-real_r)
                    onex.append(xc+gaps)
                    oney.append(yc-freal_r)
                elif place == "B":
                    twox.append(xc+gaps)
                    twoy.append(yc+real_r)
                    onex.append(xc+gaps)
                    oney.append(yc+freal_r)
    return [twox,twoy],[onex,oney]
two,one=sengiri_fissec_edges(sun_image,limb_wigth=345)

plt.imshow(cv2.imread(sun_image, cv2.IMREAD_UNCHANGED), cmap='gray')
size=25
plt.scatter(two[0], two[1], color='cyan', label='twice diff max', s=size)
plt.scatter(one[0], one[1], color='magenta', label='first diff max', s=size)
plt.legend()    
plt.show()
print ("data:", sun_image)