# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 22:33:06 2024

@author: astro
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 
from matplotlib.lines import Line2D
import networkx as nx
import cv2
import os

if __name__ == "__main__":
    idx_get_images = True
    idx_get_gif = True
    this = np.load("data/this.npy")
    Xjcall_his = np.load("data/Xjcall_his.npy")
    Zj2dall_his = np.load("data/Zj2dall_his.npy")
    res_jc_his = np.load("data/res_jc_his.npy")
    Aadhis = np.load("data/Aadhis.npy")
    m1jc_his = np.load("data/m1jc_his.npy")
    wjc_his = np.load("data/wjc_his.npy")
    Nsc = Aadhis.shape[0]
    Nint = this.shape[0]-1
    
    # figures
    Xchis = Xjcall_his[:,6*Nsc:6*(Nsc+1)]
    Xjallhis = Xjcall_his[:,0:6*Nsc]
    x3dhis = Xchis[:,0]*np.cos(Xchis[:,5])
    y3dhis = Xchis[:,0]*np.sin(Xchis[:,5])
    
    # chief residual filter
    eps_res = 5
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax1.plot(this[0:Nint],res_jc_his[Nsc,:],"C0")
    ax2.plot(this[0:Nint],m1jc_his[Nsc,:],"C1",linestyle="--")
    ax3.plot(this[0:Nint],wjc_his[Nsc,:],"C2",linestyle="--")
    ax2.legend(["fault signal","disturbance signal"])
    ax1.set_xlabel("time")
    ax1.set_ylabel("residual",color="C0")
    ax2.set_ylabel("fault signal",color="C1")
    ax3.set_ylabel("disturbance signal",color="C2")
    ax1.tick_params(axis='y',labelcolor="C0")
    ax2.tick_params(axis='y',labelcolor="C1")
    ax3.tick_params(axis='y',labelcolor="C2")
    ax3.spines["right"].set_position(("outward",60))
    plt.title("FDI filter of chief S/C")
    plt.figure()
    plt.plot(this[0:Nint],res_jc_his[Nsc,:]>=eps_res,"C0")
    plt.title("FDI filter of chief S/C")
    
    # deputy residual filter
    eps_res = 0.005
    for j in range(Nsc):
        fig,ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        ax1.plot(this[0:Nint],res_jc_his[j,:],"C0")
        ax2.plot(this[0:Nint],m1jc_his[j,:],"C1",linestyle="--")
        ax3.plot(this[0:Nint],wjc_his[j,:],"C2",linestyle="--")
        ax2.legend(["fault signal","disturbance signal"])
        ax1.set_xlabel("time")
        ax1.set_ylabel("residual",color="C0")
        ax2.set_ylabel("fault signal",color="C1")
        ax3.set_ylabel("disturbance signal",color="C2")
        ax1.tick_params(axis='y',labelcolor="C0")
        ax2.tick_params(axis='y',labelcolor="C1")
        ax3.tick_params(axis='y',labelcolor="C2")
        ax3.spines["right"].set_position(("outward",60))
        plt.title("FDI filter of deputy S/C #"+str(j+1))
        #plt.figure()
        #plt.plot(this[0:Nint],res_jc_his[j,:]>=eps_res,"C0")
        #plt.title("FDI filter of deputy S/C #"+str(j+1))
        
    # position plots
    plt.figure()
    plt.plot(x3dhis,y3dhis)
    plt.grid()
    plt.xlabel("horizontal position (km)")
    plt.ylabel("vertical position (km)")
    plt.title("absolute position of deputy S/C")
    plt.figure()
    for j in range(Nsc):
        Xjhis = Xjallhis[:,6*j:6*(j+1)]
        plt.plot(Xjhis[:,0],Xjhis[:,1])
    plt.grid()
    plt.xlabel("horizontal position (km)")
    plt.ylabel("vertical position (km)")
    plt.title("relative position of deputy S/C")
    
    # network plots
    if idx_get_images == True:
        clnominal = "#dddddd"
        cldetected = "#ffffcc"
        clisolated = "#ccaaff"
        clfdied = "#ee9999"
        clfailed = "#deaa60"
        clnormal = "#2b2b2b"
        cldamaged = "#626000"
        clperturbed = "#5400d0"
        clboth = "#841616"
        cledge = "#dddddd"
        cl_fdi_status = [clnominal,cldetected,clisolated,clfdied,clfailed]
        cl_sc_status = [clnormal,cldamaged,clperturbed,clboth]
        #p1 = mpatches.Patch(color=clnominal,label="nominal",linewidth=1) 
        p2 = Line2D([0],[0],marker="o",color=cledge,label="detected",markerfacecolor=cldetected,markersize=10)
        p3 = Line2D([0],[0],marker="o",color=cledge,label="isolated",markerfacecolor=clisolated,markersize=10)
        p4 = Line2D([0],[0],marker="o",color=cledge,label="detected & isolated",markerfacecolor=clfdied,markersize=10)
        p5 = Line2D([0],[0],marker="o",color=cledge,label="failed",markerfacecolor=clfailed,markersize=10)
        p6 = mpatches.Patch(color=clnormal,label="S/C w/o falut",linewidth=1) 
        p7 = mpatches.Patch(color=cldamaged,label="S/C w/ fault",linewidth=1) 
        p8 = mpatches.Patch(color=clperturbed,label="S/C w/ disturbance",linewidth=1) 
        p9 = mpatches.Patch(color=clboth,label="S/C w/ fault & disturbance",linewidth=1)
        patches_all = [p2,p3,p4,p5,p6,p7,p8,p9]
        ndigits = 5
        eps_ress = 0.005*np.ones(Nsc+1)
        eps_ress[Nsc] = 5
        idx_fdi_failed = np.zeros((Nsc+1,Nint))
        for k in range(Nint):
            if np.remainder(k,100) == 0:
                print("images: k =",k)
            Gk = nx.Graph()
            posGk = {}
            Gc = nx.Graph()
            posGc = {}
            color_map_node = []
            color_map_font = {}
            Aadk = Aadhis[:,:,k]
            uAadk = np.triu(Aadk)
            for j in range(Nsc+1):
                m1j = m1jc_his[j,k]
                wj = wjc_his[j,k]
                resj = res_jc_his[j,k]
                Xjhis = Xjcall_his[:,6*j:6*(j+1)]
                eps_res = eps_ress[j]
                if (m1j == 0) and (wj == 0):
                    idx_sc = 0
                    if resj < eps_res:
                        idx_fdi = 0
                    else:
                        idx_fdi = 4
                elif (m1j != 0) and (wj == 0):
                    idx_sc = 1
                    if resj < eps_res:
                        idx_fdi = 4
                    else:
                        idx_fdi = 1
                elif (m1j == 0) and (wj != 0):
                    idx_sc = 2
                    if resj < eps_res:
                        idx_fdi = 2
                    else:
                        idx_fdi = 4
                elif (m1j != 0) and (wj != 0):
                    idx_sc = 3
                    if resj < eps_res:
                        idx_fdi = 4
                    else:
                        idx_fdi = 3
                if idx_fdi == 4:
                    idx_fdi_failed[j,k] = 1
                if j < Nsc:
                    Gk.add_node(j+1)
                    posGk[j+1] = Xjhis[k,0:2]
                    color_map_node.append(cl_fdi_status[idx_fdi])
                    color_map_font[j+1] = cl_sc_status[idx_sc]
                    for i in range(Nsc):
                        aji = uAadk[j,i]
                        if aji == 1:
                            Gk.add_edge(j+1,i+1)
                else:
                    Gc.add_node(1)
                    posGc[1] = np.array([Xjhis[k,0]*np.cos(Xjhis[k,5]),Xjhis[k,0]*np.sin(Xjhis[k,5])])
                    color_node_c = cl_fdi_status[idx_fdi]
                    color_font_c = cl_sc_status[idx_sc]
            # deputy S/C
            color_map_node_rev = []
            for ni in list(Gk):
                color_map_node_rev.append(color_map_node[ni-1])
            fig = plt.figure()
            for node,color in color_map_font.items():
                nx.draw_networkx_labels(Gk,posGk,labels={node:node},font_color=color,font_weight="bold",font_size=10)
            nx.draw(Gk,posGk,with_labels=False,edge_color=cledge,width=1,node_color=color_map_node_rev,node_size=200)
            plt.legend(handles=patches_all,labelcolor="#000000",fontsize=8,facecolor=cledge,framealpha=1,loc="center left")
            plt.xlim([-20,7])
            plt.ylim([-7,7])
            fig.set_facecolor(clnormal)
            #plt.show()
            plt.savefig("figs/temp/deputy/frame"+str(k).zfill(ndigits)+".png")
            #plt.savefig("figs/temp/deputy/frame"+str(k).zfill(ndigits)+".png",dpi=300) # high quality
            plt.close()
            # chief S/C
            fig = plt.figure()
            nx.draw(Gc,posGc,with_labels=False,edge_color=cledge,width=5,node_color=color_node_c,node_size=600)
            nx.draw_networkx_labels(Gc,posGc,labels={1:"C"},font_color=color_font_c,font_weight="bold",font_size=18)
            plt.legend(handles=patches_all,labelcolor="#000000",fontsize=8,facecolor=cledge,framealpha=1,loc="lower left")
            plt.xlim([2000,5000])
            plt.ylim([-1000,4000])
            fig.set_facecolor(clnormal)
            #plt.show()
            plt.savefig("figs/temp/chief/framec"+str(k).zfill(ndigits)+".png")
            #plt.savefig("figs/temp/chief/framec"+str(k).zfill(ndigits)+".png",dpi=300) # high quality
            plt.close()
        idx_fdi_d_failed = idx_fdi_failed[0:Nsc,:]
        idx_fdi_c_failed = idx_fdi_failed[Nsc,:]
        np.save("data/idx_fdi_d_failed.npy",idx_fdi_d_failed)
        np.save("data/idx_fdi_c_failed.npy",idx_fdi_c_failed)
    if idx_get_gif == True:
        frames = []
        frames_c = []
        idx_fdi_d_failed = np.load("data/idx_fdi_d_failed.npy")
        idx_fdi_c_failed = np.load("data/idx_fdi_c_failed.npy")
        all_fdi_d = idx_fdi_d_failed.shape[0]*idx_fdi_d_failed.shape[1]
        all_fdi_c = idx_fdi_c_failed.shape[0]
        percent_success_d = (1-np.sum(idx_fdi_d_failed)/all_fdi_d)*100
        percent_success_c = (1-np.sum(idx_fdi_c_failed)/all_fdi_c)*100
        imaged_folder = "figs/temp/deputy"
        imagec_folder = "figs/temp/chief"
        videod_name = "figs/deputyscfdi.mp4"
        videoc_name = "figs/chiefscfdi.mp4"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 0.5
        font_color = (221,221,221)
        font_thickness = 1
        textd = "FDI success rate: "+str(round(percent_success_d,1))+"%"
        textc = "FDI success rate: "+str(round(percent_success_c,1))+"%"
        xt,yt = 50,50
        imagesd = [img for img in os.listdir(imaged_folder) if img.endswith(".png")]
        imagesc = [img for img in os.listdir(imagec_folder) if img.endswith(".png")]
        framed = cv2.imread(os.path.join(imaged_folder,imagesd[0]))
        framec = cv2.imread(os.path.join(imagec_folder,imagesc[0]))
        heightd,widthd,layersd = framed.shape
        heightc,widthc,layersc = framec.shape
        videod = cv2.VideoWriter(videod_name,0,100,(widthd,heightd))    
        videoc = cv2.VideoWriter(videoc_name,0,100,(widthc,heightc))    
        for image in imagesd:
            img = cv2.imread(os.path.join(imaged_folder,image))
            img = cv2.putText(img,textd,(xt,yt),font,font_size,font_color,font_thickness,cv2.LINE_AA)
            videod.write(img)
        for image in imagesc:
            img = cv2.imread(os.path.join(imagec_folder,image))
            img = cv2.putText(img,textc,(xt,yt),font,font_size,font_color,font_thickness,cv2.LINE_AA)
            videoc.write(img)
        cv2.destroyAllWindows()
        videod.release()
        videoc.release()