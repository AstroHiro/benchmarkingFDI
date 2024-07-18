# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 02:28:58 2024

@author: astro
"""

import numpy as np
import scipy as sp
import cvxpy as cp
import matplotlib.pyplot as plt
import control

class Benchmark():
    def __init__(self,mu,Nsc=1):
        self.mu = mu # gravitational constant
        self.Nsc = Nsc # number of S/C
        self.kJ2 = 0 #self.kJ2 = 2.6330e+10 # J2 coefficient (of Earth)
        self.ome = 7.2921e-5 # rotation speed (of Earth)
        self.Cd = 0 #self.Cd = 1 # drag coefficient of chief S/C
        self.mass = 1 # mass of chief S/C
        self.rho = 1e-8 # air density for chief S/C
        self.Area = 1 # cross-sectional area of chief S/C
        self.Cdj = self.Cd # drag coefficient of deputy S/C
        self.massj = self.mass # mass of deputy S/C
        self.rhoj = self.rho # air density for deputy S/C
        self.Areaj = self.Area # cross-sectional area of deputy S/C
        self.h = 0.1 # step size for numerical integration
        self.Lamdd = 1*np.identity(3) # sliding mode control gain
        self.K1 = np.identity(3)*0.5 # tracking control gain
        self.K2 = np.identity(3)*0.02 # synchronization control gain
        self.nX = 6 # number of states
        self.nP = 3 # number of positions
        self.nU = 3 # number of control inputs
        self.nX2d = 4 # number of 2d states
        self.nP2d = 2 # number of 2d positions
        self.nU2d = 2 # number of 2d control inputs
        self.nY = 3 # number of measurements
        self.Nlinks = 1 # initial number of network communication links
        self.getparams()
        self.changecommunication()
    
    ##########################
    ##### initialization #####
    ##########################
    
    def getparams(self):
        # compute parameters for benchmark problem
        p = self.Nsc
        nP = self.nP
        n_u = nP*p
        xe = 2
        ye = 2
        ze = 0.
        pe0 = np.deg2rad(0.573)
        pz0 = np.deg2rad(11.46)
        nphase = 0.0011
        qd = lambda t: np.array([xe*np.sin(nphase*t+pe0),ye*np.cos(nphase*t+pe0),ze*np.sin(nphase*t+pz0)])
        qd_dot = lambda t: nphase*np.array([xe*np.cos(nphase*t+pe0),-ye*np.sin(nphase*t+pe0),ze*np.cos(nphase*t+pz0)])
        qd_ddot = lambda t: -nphase**2*np.array([xe*np.sin(nphase*t+pe0),ye*np.cos(nphase*t+pe0),ze*np.sin(nphase*t+pz0)])
        Rfd,Tall,idx_targets = self.gettransmats(xe,ye,ze,pe0,pz0)
        Qd = lambda t: np.array([qd(t),qd_dot(t),qd_ddot(t)])
        d0 = 0.2
        q0_all = np.zeros(n_u)
        q0_dot_all = np.zeros(n_u)
        for j in range(p):
            xj0 = -d0+2*d0*np.random.rand(1)[0]
            yj0 = -d0+2*d0*np.random.rand(1)[0]
            zj0 = -d0+2*d0*np.random.rand(1)[0]
            q0_all[nP*j:nP*(j+1)] = np.array([xj0,yj0,zj0])
        self.Qdfun = Qd
        self.q0_all = q0_all
        self.q0_dot_all = q0_dot_all
        self.Rfd = Rfd 
        self.R_d = np.kron(np.identity(p),Rfd)
        self.Tall = Tall
        self.idx_targets = idx_targets       
        Rq2X = np.zeros((2*p,p))
        Rv2X = np.zeros((2*p,p))
        for j in range(p):
            Rq2X[2*j,j] = 1
            Rv2X[2*(j+1)-1,j] = 1
        Rq2X = np.kron(Rq2X,np.identity(nP))
        Rv2X = np.kron(Rv2X,np.identity(nP))
        Rqq2X = np.hstack((Rq2X,Rv2X))
        self.Rqq2X = Rqq2X
        R3d22dj = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0]])
        self.R3d22d = np.kron(np.identity(p),R3d22dj)
        Ru3d2u2dj = np.array([[1,0,0],[0,1,0]])
        self.Ru3d2u2d = np.kron(np.identity(p),Ru3d2u2dj)
        self.Xjall0 = Rqq2X@np.hstack((q0_all,q0_dot_all))            
        self.Bj2d = np.array([[0,0],[0,0],[1,0],[0,1]])
        self.Pj2d = np.array([[0],[0],[0],[1]])
        self.Lj2d = np.array([[0],[0],[1],[0]])
        self.Cj2d = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1]])
        self.B2dall = np.kron(np.identity(p),self.Bj2d)
        self.P2dall = np.kron(np.identity(p),self.Pj2d)
        self.L2dall = np.kron(np.identity(p),self.Lj2d)
        CP = self.Cj2d@self.Pj2d
        self.Nj2d = self.Pj2d@np.linalg.pinv((CP).T@CP)@CP.T
        pass
    
    def gettransmats(self,xe,ye,ze,psie0,psiz0):
        # compute transformation matrices for target trajectories
        p = self.Nsc
        nP = self.nP
        lmin,lmax,PHI = self.getmaxes(xe,ye,ze,psie0,psiz0)
        R11 = -xe/lmin*np.sin(PHI-psie0)
        R12 = ye/lmin*np.cos(PHI-psie0)
        R13 = -ze/lmin*np.sin(PHI-psiz0)
        R21 = -xe/lmax*np.cos(PHI-psie0)
        R22 = -ye/lmax*np.sin(PHI-psie0)
        R23 = -ze/lmax*np.cos(PHI-psiz0)
        R31 = -ye*ze/lmin/lmax*np.cos(psie0-psiz0)
        R32 = xe*ze/lmin/lmax*np.sin(psie0-psiz0)
        R33 = xe*ye/lmin/lmax
        R1d = np.array([[R11,R12,R13],[R21,R22,R23],[R31,R32,R33]])
        R2d = np.array([[1,0,0],[0,lmin/lmax,0],[0,0,1]])
        Rfd = R2d@R1d
        Tall = np.zeros((nP*p,nP*p))
        idx = 1
        nidx = 4
        phi0 = 0
        dphi = 0
        for j in range(p):
            if j > nidx/2*idx*(idx+1):
                phi0 = phi0+dphi
                idx = idx+1
            sat_num = nidx*idx
            phi = 2*np.pi/sat_num
            Tz = np.array([[np.cos((j-1)*phi+phi0),-np.sin((j-1)*phi+phi0)],[np.sin((j-1)*phi+phi0),np.cos((j-1)*phi+phi0)]])
            Tz = idx*np.identity(2)@Tz
            Tall[nP*j:nP*(j+1),nP*j:nP*(j+1)] = np.vstack((np.hstack((Tz,np.zeros((2,1)))),np.hstack((np.zeros(2),1))))
        idx_targets = idx
        return Rfd,Tall,idx_targets
    
    def getmaxes(self,xe,ye,ze,psie0,psiz0):
        # compute parameters for nominal target trajectories
        numP = (xe**2-ye**2)*np.sin(2*psie0)+ze**2*np.sin(2*psiz0)
        denP = (xe**2-ye**2)*np.cos(2*psie0)+ze**2*np.cos(2*psiz0)
        PHI = 1/2*np.arctan2(numP,denP)
        xdfun = lambda psi: np.array([xe*np.sin(psi+psie0),ye*np.cos(psi+psie0),ze*np.sin(psi+psiz0)])
        lfun = lambda psi: np.linalg.norm(xdfun(psi))
        lmin = lfun(-PHI)
        lmax = lfun(3/2*np.pi-PHI)
        return lmin,lmax,PHI
    
    def changecommunication(self):
        # change communication links for multi-S/C network
        idx_error = 100 # large number
        counter1 = 0
        while idx_error >= 1e-8:
            Nlinks = 1
            p = self.Nsc
            nP = self.nP
            d_rand = 5
            pos_rand = -d_rand+2*np.random.rand(p*nP)
            pos_mat = pos_rand.reshape(p,nP)
            dists_rand = sp.spatial.distance.pdist(pos_mat)
            dists_rand_sq = sp.spatial.distance.squareform(dists_rand)
            dists_rand_tr = np.triu(dists_rand_sq)
            dists_rand_tr[dists_rand_tr == 0] = 100 # large number
            dists = dists_rand_tr.reshape(p*p)
            idxdists = np.argsort(dists)
            Nlinks_out = Nlinks
            idxp = 0
            counter2 = 0
            while idxp != p:
                vecAad = np.zeros(p*p)
                vecAad[idxdists[0:Nlinks_out]] = 1
                uAad = vecAad.reshape(p,p)
                Aad = uAad+uAad.T
                Aadp = np.linalg.matrix_power(Aad,p)
                Nlinks_out = Nlinks_out+1
                idxp = max(sum(Aadp != 0))
                counter2 += 1 
                if counter2 == p*(p-1)/2:
                    self.Aadfail = Aad
                    self.Nlinksfail = Nlinks_out
                    raise ValueError("invalid adjacency matrix")
            Nlinks_out = Nlinks_out-1
            self.Aad = Aad
            self.Nlinks = Nlinks_out
            self.C2dall = self.getCmatrix(Aad)
            C0 = np.kron(np.identity(p),self.Cj2d)
            self.Ctrans = C0@np.linalg.pinv(self.C2dall)
            idx_error = np.linalg.norm(self.Ctrans@self.C2dall-C0)
            CP = self.Ctrans@self.C2dall@self.P2dall
            self.N2dall = self.P2dall@np.linalg.pinv((CP).T@CP)@CP.T
            counter1 += 1
            if counter1 == 100:
                raise ValueError("invalid network topology")
        pass
    
    def getCmatrix(self,Aad):
        # compute system output matrix
        p = self.Nsc
        Cj2d = self.Cj2d
        cchief = np.zeros(p)
        cchief[0] = 1
        uAad = np.triu(Aad)
        idx_nzeroX,idx_nzeroY = np.nonzero(uAad)
        Nnzero = idx_nzeroX.shape[0]
        ncrow = np.sum(Nnzero)+1
        call = np.zeros((ncrow,p))
        for k in range(Nnzero):
            iX = idx_nzeroX[k]
            iY = idx_nzeroY[k]
            call[k,iX] = 1
            call[k,iY] = -1
        call[Nnzero,:] = cchief
        C2dall = np.kron(call,Cj2d)
        return C2dall
    
    ##########################
    #### dynamical system ####
    ##########################    
    
    def dynamicspolar3d(self,t,X,U,m1,w):
        # compute dynamics of chief S/C
        mu = self.mu
        kJ2 = self.kJ2
        ome = self.ome
        Cd = self.Cd
        mass = self.mass
        rho = self.rho
        Area = self.Area
        uR = U[0]
        uT = U[1]
        uN = U[2]
        r = X[0]
        vx = X[1]
        h = X[2]
        i = X[4]
        th = X[5]
        si = np.sin(i)
        ci = np.cos(i)
        s2i = np.sin(2*i)
        sth = np.sin(th)
        cth = np.cos(th)
        s2th = np.sin(2*th)
        C = Cd*(Area/mass)*rho/2
        Va = np.array([vx,h/r-ome*r*ci,ome*r*cth*si])
        va = np.linalg.norm(Va)
        drdt = vx
        dvxdt = -mu/r**2+h**2/r**3-kJ2/r**4*(1-3*si**2*sth**2)-C*va*vx+uR+w
        dhdt = -kJ2*si**2*s2th/r**3-C*va*(h-ome*r**2*ci)+r*uT+r*m1
        dOmdt = -2*kJ2*ci*sth**2/h/r**3-C*va*ome*r**2*s2th/2/h+(r*sth/h/si)*uN
        didt = -kJ2*s2i*s2th/2/h/r**3-C*va*ome*r**2*si*cth**2/h+(r*cth/h)*uN
        dthdt = h/r**2+2*kJ2*ci**2*sth**2/h/r**3+C*va*ome*r**2*ci*s2th/2/h-(r*sth*ci/h/si)*uN
        dXdt = np.array([drdt,dvxdt,dhdt,dOmdt,didt,dthdt])
        return dXdt
    
    def dynamicsrelative(self,t,Xj,Xc,Uj,Uc,m1c,wc,m1j,wj):
        # compute relative dynamics of deputy S/C
        mu = self.mu
        kJ2 = self.kJ2
        ome = self.ome
        Cd = self.Cd
        mass = self.mass
        rho = self.rho
        Area = self.Area
        Cdj = self.Cdj
        massj = self.massj
        rhoj = self.rhoj
        Areaj = self.Areaj
        nX = self.nX
        nP = self.nP
        C = Cd*(Area/mass)*rho/2
        Cj = Cdj*(Areaj/massj)*rhoj/2
        xj = Xj[0]
        yj = Xj[1]
        zj = Xj[2]
        xjdot = Xj[3]
        yjdot = Xj[4]
        zjdot = Xj[5]
        r = Xc[0]
        vx = Xc[1]
        h = Xc[2]
        i = Xc[4]
        th = Xc[5]
        uR = Uc[0]
        uT = Uc[1]
        uN = Uc[2]
        dXcdt = self.dynamicspolar3d(t,Xc,Uc,m1c,wc)
        Omdot = dXcdt[3]
        idot = dXcdt[4]
        thdot = dXcdt[5]
        si = np.sin(i)
        ci = np.cos(i)
        s2i = np.sin(2*i)
        sth = np.sin(th)
        cth = np.cos(th)
        s2th = np.sin(2*th)
        lj = Xj[0:nP]
        ljdot = Xj[nP:nX]
        Va = np.array([vx,h/r-ome*r*ci,ome*r*cth*si])
        va = np.linalg.norm(Va)
        omx = idot*cth+Omdot*sth*si+r/h*uN
        omz = thdot+Omdot*ci
        om_vec = np.array([omx,0,omz])
        Vaj = Va+ljdot+np.cross(om_vec,lj)
        vaj = np.linalg.norm(Vaj)
        rj = np.sqrt((r+xj)**2+yj**2+zj**2)
        rjZ = (r+xj)*si*sth+yj*si*cth+zj*ci
        zt = 2*kJ2*si*sth/r**4
        ztj = 2*kJ2*rjZ/rj**5
        alx = -kJ2*s2i*cth/r**5+3*vx*kJ2*s2i*sth/r**4/h-8*kJ2**2*si**3*ci*sth**2*cth/r**6/h**2
        alz = -2*h*vx/r**3-kJ2*si**2*s2th/r**5
        et = np.sqrt(mu/r**3+kJ2/r**5-5*kJ2*si**2*sth**2/r**5)
        etj = np.sqrt(mu/rj**3+kJ2/rj**5-5*kJ2*rjZ**2/rj**7)
        xjddot = 2*yjdot*omz-xj*(etj**2-omz**2)+yj*alz-zj*omx*omz-(ztj-zt)*si*sth-r*(etj**2-et**2)-Cj*vaj*(xjdot-yj*omz)-(Cj*vaj-C*va)*vx-(uR+wc)
        yjddot = -2*xjdot*omz+2*zjdot*omx-xj*alz-yj*(etj**2-omz**2-omx**2)+zj*alx-(ztj-zt)*si*cth-Cj*vaj*(yjdot+xj*omz-zj*omx)-(Cj*vaj-C*va)*(h/r-ome*r*ci)-(uT+m1c)
        zjddot = -2*yjdot*omx-xj*omx*omz-yj*alx-zj*(etj**2-omx**2)-(ztj-zt)*ci-Cj*vaj*(zjdot+yj*omx)-(Cj*vaj-C*va)*ome*r*cth*si-uN
        eyeq = np.identity(3)
        Zj = np.zeros((3,3))
        Bj = np.vstack((Zj,eyeq))
        Lj = np.array([0,0,0,1,0,0])
        Pj = np.array([0,0,0,0,1,0])
        dXjdt = np.array([xjdot,yjdot,zjdot,xjddot,yjddot,zjddot])+Bj@Uj+Lj*m1j+Pj*wj
        return dXjdt

    def dynamicsall(self,t,Xjcall,Ujcall,m1jcall,wjcall):
        # compute stacked dynamics of chief and deputy S/C
        p = self.Nsc
        nX = self.nX
        nU = self.nU
        Xjall = Xjcall[0:nX*p]
        Xc = Xjcall[nX*p:nX*(p+1)]
        Ujall = Ujcall[0:nU*p]
        Uc = Ujcall[nU*p:nU*(p+1)]
        m1c = m1jcall[p]
        wc = wjcall[p]
        dXdcalldt = np.zeros_like(Xjcall)
        for j in range(p):
            Xj = Xjall[nX*j:nX*(j+1)]
            Uj = Ujall[nU*j:nU*(j+1)]
            m1j = m1jcall[j]
            wj = wjcall[j]
            dXdcalldt[nX*j:nX*(j+1)] = self.dynamicsrelative(t,Xj,Xc,Uj,Uc,m1c,wc,m1j,wj)
        dXdcalldt[nX*p:nX*(p+1)] = self.dynamicspolar3d(t,Xc,Uc,m1c,wc)
        return dXdcalldt
    
    def measurementXc(self,t,Xc):
        # compute system output of chief S/C state
        r = Xc[0]
        h = Xc[2]
        th = Xc[5]
        Yc = np.array([r,h,th])
        return Yc
    
    def measurementnet(self,t,Xjall):
        # compute system output of deputy S/C state
        Xj2dall = self.R3d22d@Xjall
        Yjall = self.C2dall@Xj2dall
        return Yjall
    
    ##########################
    ####### controller #######
    ##########################
    
    def expcontrol(self,t,Xjall,Xc,Uc,Qd_t):
        # compute nominal control input of deputy S/C
        Lap = self.getlaplacian(self.K1,self.K2)
        R_d = self.R_d
        M,C,G,D,dXcdt = self.lagrangeMCGD(t,Xjall,Xc,Uc)
        M_d = R_d@M@R_d.T
        C_d = R_d@C@R_d.T
        G_d = R_d@G
        D_d = R_d@D
        s_d,qr_dot_d,qr_ddot_d = self.getsd(Xjall,Qd_t)
        Rfdtau = M_d@qr_ddot_d+C_d@qr_dot_d+G_d+D_d-Lap@s_d
        U = R_d.T@Rfdtau
        return U

    def lagrangeMCGD(self,t,Xjall,Xc,Uc):
        # compute lagrangian dynamics of all deputy S/C
        p = self.Nsc
        nX = self.nX
        nP = self.nP
        M = np.zeros((nP*p,nP*p))
        C = np.zeros((nP*p,nP*p))
        G = np.zeros(nP*p)
        D = np.zeros(nP*p)
        for j in range(p):
            Xj = Xjall[nX*j:nX*(j+1)]
            Mj,Cj,Gj,Dj,dXcdt = self.lagrangeMCGDj(t,Xj,Xc,Uc)
            M[nP*j:nP*(j+1),nP*j:nP*(j+1)] = Mj
            C[nP*j:nP*(j+1),nP*j:nP*(j+1)] = Cj
            G[nP*j:nP*(j+1)] = Gj
            D[nP*j:nP*(j+1)] = Dj
        return M,C,G,D,dXcdt
    
    def lagrangeMCGDj(self,t,Xj,Xc,Uc):
        # compute lagrangian dynamics of deputy S/C
        mu = self.mu
        kJ2 = self.kJ2
        ome = self.ome
        Cd = self.Cd
        mass = self.mass
        rho = self.rho
        Area = self.Area
        Cdj = self.Cdj
        massj = self.massj
        rhoj = self.rhoj
        Areaj = self.Areaj
        nX = self.nX
        nP = self.nP
        C = Cd*(Area/mass)*rho/2
        Cj = Cdj*(Areaj/massj)*rhoj/2
        xj = Xj[0]
        yj = Xj[1]
        zj = Xj[2]
        xjdot = Xj[3]
        yjdot = Xj[4]
        zjdot = Xj[5]
        r = Xc[0]
        vx = Xc[1]
        h = Xc[2]
        i = Xc[4]
        th = Xc[5]
        uR = Uc[0]
        uT = Uc[1]
        uN = Uc[2]
        dXcdt = self.dynamicspolar3d(t,Xc,Uc,0,0)
        Omdot = dXcdt[3]
        idot = dXcdt[4]
        thdot = dXcdt[5]
        si = np.sin(i)
        ci = np.cos(i)
        s2i = np.sin(2*i)
        sth = np.sin(th)
        cth = np.cos(th)
        s2th = np.sin(2*th)
        lj = Xj[0:nP]
        ljdot = Xj[nP:nX]
        Va = np.array([vx,h/r-ome*r*ci,ome*r*cth*si])
        va = np.linalg.norm(Va)
        omx = idot*cth+Omdot*sth*si+r/h*uN
        omz = thdot+Omdot*ci
        om_vec = np.array([omx,0,omz])
        Vaj = Va+ljdot+np.cross(om_vec,lj)
        vaj = np.linalg.norm(Vaj)
        rj = np.sqrt((r+xj)**2+yj**2+zj**2)
        rjZ = (r+xj)*si*sth+yj*si*cth+zj*ci
        zt = 2*kJ2*si*sth/r**4
        ztj = 2*kJ2*rjZ/rj**5
        alx = -kJ2*s2i*cth/r**5+3*vx*kJ2*s2i*sth/r**4/h-8*kJ2**2*si**3*ci*sth**2*cth/r**6/h**2
        alz = -2*h*vx/r**3-kJ2*si**2*s2th/r**5
        et = np.sqrt(mu/r**3+kJ2/r**5-5*kJ2*si**2*sth**2/r**5)
        etj = np.sqrt(mu/rj**3+kJ2/rj**5-5*kJ2*rjZ**2/rj**7)
        Mmat = np.identity(3)
        Cmat = -2*np.array([[0,omz,0],[-omz,0,omx],[0,-omx,0]])
        Gmat = np.array([[etj**2-omz**2,-alz,omx*omz],[alz,etj**2-omz**2-omx**2,-alx],[omx*omz,alx,etj**2-omx**2]])@Xj[0:3]+(ztj-zt)*np.array([si*sth,si*cth,ci])+np.array([r*(etj**2-et**2),0,0])
        Dmat = -np.array([-Cj*vaj*(xjdot-yj*omz)-(Cj*vaj-C*va)*vx-uR,-Cj*vaj*(yjdot+xj*omz-zj*omx)-(Cj*vaj-C*va)*(h/r-ome*r*ci)-uT,-Cj*vaj*(zjdot+yj*omx)-(Cj*vaj-C*va)*ome*r*cth*si-uN])
        return Mmat,Cmat,Gmat,Dmat,dXcdt
    
    def getsd(self,Xjall,Qd_t):
        # compute composite state of deputy S/C
        nX = self.nX
        nP = self.nP
        p = self.Nsc
        Tall = self.Tall
        Lamdd = self.Lamdd
        qq = self.Rqq2X.T@Xjall
        q_all = qq[0:nP*p]
        q_dot_all = qq[nP*p:nX*p]
        qd = Qd_t[0,:]
        qd_dot = Qd_t[1,:]
        qd_ddot = Qd_t[2,:]
        R_d = self.R_d
        Lam_all = np.kron(np.identity(p),Lamdd)
        qd_all = np.tile(qd,p)
        qd_dot_all = np.tile(qd_dot,p)
        qd_ddot_all = np.tile(qd_ddot,p)
        q_d = R_d@q_all
        q_dot_d = R_d@q_dot_all
        qd_d = R_d@qd_all
        qd_dot_d = R_d@qd_dot_all
        qd_ddot_d = R_d@qd_ddot_all
        qr_dot_d = Tall@qd_dot_d-Lam_all@(q_d-Tall@qd_d)
        qr_ddot_d = Tall@qd_ddot_d-Lam_all@(q_dot_d-Tall@qd_dot_d)
        s_d = q_dot_d-qr_dot_d
        return s_d,qr_dot_d,qr_ddot_d
    
    def getlaplacian(self,M1,M2):
        # compute Laplacian matrix of multi-S/C network
        p = self.Nsc
        Aad = self.Aad
        L = np.zeros((3*p,3*p))
        I = np.identity(p)
        L = np.kron(I,M1)-np.kron(Aad,M2)
        return L
    
    ##########################
    ####### FDI filter #######
    ##########################
    
    def residualfilterXc(self,t,xi_c,Yc,Uc):
        # compute dynamics of chief FDI filter
        h = Yc[1]
        r = Yc[0]
        u2 = Uc[1]
        k = 10
        dxi_c_dt = r*u2+k*(h-xi_c)
        return dxi_c_dt

    def residualfilternet(self,t,Zj2dall,Yjall,Ujall,Xc,Uc,m1c,wc):
        # compute dynamics of deputy FDI filter
        B = self.B2dall
        C = self.Ctrans@self.C2dall
        N = self.N2dall
        nX = Zj2dall.shape[0]
        Yjall_t = self.Ctrans@Yjall
        nY = Yjall_t.shape[0]
        Xhatj2dall = Zj2dall+N@Yjall_t
        phifun = lambda X: self.residualphinet(t,X,Xc,Uc,m1c,wc)
        phi = phifun(Xhatj2dall)
        Aphi = self.getdfdX(Xhatj2dall,phifun)
        I = np.identity(nX)
        M = I-N@C
        G = M@B
        Aekf = M@Aphi
        Q = np.identity(nX)*10
        R = np.identity(nY)
        KT,_,_ = control.lqr(Aekf.T,C.T,Q,R)
        K = -KT.T
        F = K@C
        H = K@C@N-K
        dZj2dall_dt = F@Zj2dall+G@self.Ru3d2u2d@Ujall+M@phi+H@Yjall_t
        return dZj2dall_dt
    
    def residualphinet(self,t,Xj2dall,Xc,Uc,m1c,wc):
        # stack unperturbed vector fields of deputy FDI filter
        p = self.Nsc
        nX2d = self.nX2d
        phinet = np.zeros_like(Xj2dall)
        for j in range(p):
            Xj2d = Xj2dall[nX2d*j:nX2d*(j+1)]
            phinet[nX2d*j:nX2d*(j+1)] = self.residualphij(t,Xj2d,Xc,Uc,m1c,wc)
        return phinet
    
    def residualphij(self,t,Xj2d,Xc,Uc,m1c,wc):
        # compute unperturbed vector field of deputy FDI filter
        mu = self.mu
        xj = Xj2d[0]
        yj = Xj2d[1]
        xjdot = Xj2d[2]
        yjdot = Xj2d[3]
        r = Xc[0]
        vx = Xc[1]
        h = Xc[2]
        uR = Uc[0]
        uT = Uc[1]
        omz = h/r**2
        rj = np.sqrt((r+xj)**2+yj**2)
        alz = -2*h*vx/r**3
        et = np.sqrt(mu/r**3)
        etj = np.sqrt(mu/rj**3)
        xjddot = 2*yjdot*omz-xj*(etj**2-omz**2)+yj*alz-r*(etj**2-et**2)-(uR+wc)
        yjddot = -2*xjdot*omz-xj*alz-yj*(etj**2-omz**2)-(uT+m1c)
        dXj2d_dt = np.array([xjdot,yjdot,xjddot,yjddot])
        return dXj2d_dt
    
    def getdfdX(self,X,fun):
        # compute partial derivative of fun
        dx = 0.001
        n = X.shape[0]
        dfdX = np.zeros((n,n))
        for i in range(n):
            ei = np.zeros(n)
            ei[i] = 1
            f1 = fun(X+ei*dx)
            f0 = fun(X-ei*dx)
            dfdx = (f1-f0)/2/dx
            dfdX[:,i] = dfdx
        return dfdX
    
    ##########################
    ### FDI(R) simulation  ###
    ##########################    
    
    def generatesignals(self,Nint,Nfaults,dist_period,fsize):
        # generate fault and disturbance signals for FDI
        p = self.Nsc
        max_int = int(Nint/(dist_period))
        idx_array = np.arange(1,max_int)
        idx_m1jc_all = np.zeros((p+1,Nfaults))
        idx_wjc_all = np.zeros((p+1,Nfaults))
        for j in range(p+1):            
            idx_m1jc = np.random.choice(idx_array,Nfaults,False)*dist_period
            idx_m1jc.sort()
            idx_wjc = np.random.choice(idx_array,Nfaults,False)*dist_period
            idx_wjc.sort()
            idx_m1jc_all[j,:] = idx_m1jc
            idx_wjc_all[j,:] = idx_wjc
        m1jc_values = -fsize+2*fsize*np.random.rand(p+1,Nfaults)
        wjc_values = -fsize+2*fsize*np.random.rand(p+1,Nfaults)
        idx_m1_signals = np.zeros(p+1,dtype=np.uint32)
        idx_w_signals = np.zeros(p+1,dtype=np.uint32)
        m1jc_his = np.zeros((p+1,Nint))
        wjc_his = np.zeros((p+1,Nint))
        for k in range(Nint):
            for j in range(p+1):
                idx_m1j = idx_m1jc_all[j,idx_m1_signals[j]]
                if idx_m1j <= k < idx_m1j+dist_period:
                    m1j = m1jc_values[j,idx_m1_signals[j]]
                else:
                    m1j = 0
                if (k == idx_m1j+dist_period) and (idx_m1_signals[j] < Nfaults-1):
                    idx_m1_signals[j] = idx_m1_signals[j]+1
                idx_wj = idx_wjc_all[j,idx_w_signals[j]]
                if idx_wj <= k < idx_wj+dist_period:
                    wj = wjc_values[j,idx_w_signals[j]]
                else:
                    wj = 0
                if (k == idx_wj+dist_period) and (idx_w_signals[j] < Nfaults-1):
                    idx_w_signals[j] = idx_w_signals[j]+1
                m1jc_his[j,k] = m1j
                wjc_his[j,k] = wj
        #for j in range(p):
        #    plt.figure()
        #    plt.plot(m1jc_all[j,:])
        return m1jc_his,wjc_his
    
    def rk4(self,t,X,U,dynamics):
        # integrate dynamics
        h = self.h
        k1 = dynamics(t,X,U)
        k2 = dynamics(t+h/2.,X+k1*h/2.,U)
        k3 = dynamics(t+h/2.,X+k2*h/2.,U)
        k4 = dynamics(t+h,X+k3*h,U)
        return t+h,X+h*(k1+2.*k2+2.*k3+k4)/6.
    
    ##########################
    ###### soft landing ######
    ##########################
    
    def dynamicsf2dscaled(self,t,X,Cs):
        Cr = Cs[0]
        Cv = Cs[1]
        Ch = Cs[2]
        Cth = Cs[3]
        r = X[0]
        v = X[1]
        h = X[2]
        f1 = (Cv/Cr)*v
        f2 = ((Ch**2/Cr**3)*h**2/r**3-(self.mu/Cr**2)/r**2)/Cv
        f3 = 0
        f4 = ((Ch/Cr**2)*h/r**2)/Cth
        f = np.array([f1,f2,f3,f4])
        return f
    
    def dynamicsB2dscaled(self,Cs):
        Cr = Cs[0]
        Cv = Cs[1]
        Ch = Cs[2]
        B = np.array([[0,0],[1/Cv,0],[0,Cr/Ch],[0,0]])
        return B
    
    def softlandingscp(self,X0,Xf):
        K = 1
        N = 1000
        nX2d = self.nX2d
        nU2d = self.nU2d
        p = self.Nsc
        r0 = X0[0]
        h0 = X0[2]
        Cv = np.sqrt(self.mu/r0)
        Cs = np.array([r0,Cv,h0,2*np.pi])
        t0 = 0
        X0_scaled = X0/Cs
        Xf_scaled = Xf/Cs
        t = t0
        X = X0_scaled
        Xini = np.zeros((N+1,nX2d))
        Xini[0,:] = X
        dynamicsf = lambda X: self.dynamicsf2dscaled(t,X,Cs)
        B = self.dynamicsB2dscaled(Cs)
        dynamics = lambda t,X,U: dynamicsf(X)+B@U
        dt = self.h
        for n in range(N):
            U = np.zeros(2)
            t,X = self.rk4(t,X,U,dynamics)
            Xini[n+1,:] = X
        Xps = Xini
        Xf_scaled[3] = X[3]
        #xini = Xini[:,0]*Cs[0]*np.cos(Xini[:,3]*Cs[3])
        #yini = Xini[:,0]*Cs[0]*np.sin(Xini[:,3]*Cs[3])
        #plt.figure()
        #plt.plot(xini,yini)
        #plt.show()
        Cp = 10000
        Tregion = 0.1
        Xss = []
        Uss = []
        dth = np.deg2rad(0.08)
        Nsc_half = p/2        
        for j in range(p+1):
            Xf_scaled[3] = Xini[N,3]+(j-Nsc_half)*dth
            Xs = cp.Variable((N+1,nX2d))
            Us = cp.Variable((N,nU2d))
            penalty = cp.Variable(1)
            constraints = []
            constraints.append(Xs[0,:] == X0_scaled)
            #constraints.append(cp.norm(Xs[N,:]-Xf_scaled) <= penalty)
            #J = Cp*penalty**2
            constraints.append(Xs[N,:] == Xf_scaled)
            J = 0
            for n in range(N):
                Xn = Xs[n,:]
                Xnp1 = Xs[n+1,:]
                Xp = Xps[n,:]
                Un = Us[n,:]
                dfdX = self.getdfdX(Xp,dynamicsf)
                constraints.append(Xnp1 == Xn+(dynamicsf(Xp)+dfdX@(Xn-Xp)+B@Un)*dt)
                #constraints.append(cp.norm(Xn-Xp) <= Tregion)
                J += cp.sum_squares(Un)
            objective = cp.Minimize(J)
            prob = cp.Problem(objective,constraints)
            result = prob.solve()
            #Xps = Xs.value
            Xss.append(Xs.value)
            Uss.append(Us.value)
            t = t0
            X = X0_scaled
            treal = np.zeros(N+1)
            Xreal = np.zeros((N+1,nX2d))
            treal[0] = t
            Xreal[0,:] = X0_scaled
            for n in range(N):
                #Ax = self.getdfdX(X,dynamicsf)
                #Xd = Xs.value[n,:]
                Ud = Us.value[n,:]
                #Q = np.identity(nX2d)*100
                #R = np.identity(nU2d)
                #K,_,_ = control.lqr(Ax,B,Q,R)
                #U = Ud-K@(X-Xd)
                U = Ud
                t,X = self.rk4(t,X,U,dynamics)
                treal[n+1] = t
                Xreal[n+1,:] = X
            rreal = Xreal[:,0]*Cs[0]
            vreal = Xreal[:,1]*Cs[1]
            hreal = Xreal[:,2]*Cs[2]
            threal = Xreal[:,3]*Cs[3]
            omreal = hreal/rreal**2
            plt.figure()
            plt.plot(treal,rreal)
            plt.figure()
            plt.plot(treal,vreal)
            plt.figure()
            plt.plot(treal,hreal)
            plt.figure()
            plt.plot(treal,threal)
            plt.show()
            plt.figure()
            plt.plot(treal,omreal)
            plt.show()
            print("penalty",penalty.value)
            print("J",J.value)
        Xdcarss = []
        thhiss = []
        plt.figure()
        for j in range(p+1):
            Xsj = Xss[j] 
            Xdcars = np.zeros((N+1,nX2d))
            rrealj = Xsj[:,0]*Cs[0]
            vrealj = Xsj[:,1]*Cs[1]
            hrealj = Xsj[:,2]*Cs[2]
            threalj = Xsj[:,3]*Cs[3]
            omrealj = hrealj/rrealj**2
            xrealj = rrealj*np.cos(threalj)
            yrealj = rrealj*np.sin(threalj)
            vxrealj = vrealj*np.cos(threalj)-rrealj*omrealj*np.sin(threalj)
            vyrealj = vrealj*np.sin(threalj)+rrealj*omrealj*np.cos(threalj)
            plt.plot(xrealj,yrealj)
            Xdcars[:,0] = xrealj
            Xdcars[:,1] = yrealj
            Xdcars[:,2] = vxrealj
            Xdcars[:,3] = vyrealj
            Xdcarss.append(Xdcars)
            thhiss.append(threalj)
        Xdcar_chief = Xdcarss[p]
        thhis_chief = thhiss[p]
        for j in range(p):
            Xdcarj = Xdcarss[j]
            Xrcar = Xdcarj-Xdcar_chief
            Xjhis = np.zeros((N+1,nX2d))
            for n in range(N+1):
                Xrcar_n = Xrcar[n,:]
                thn = thhis_chief[n]
                Rot = np.array([[np.cos(thn),np.sin(thn)],[-np.sin(thn),np.cos(thn)]])
                Xr = np.kron(np.identity(2),Rot)@Xrcar_n
                #Xjhis[]
        plt.show()
        return Xs    
    
    def rad2car(self,vrad,th):
        # express vrad in cartesian coordinate
        # er = ex*np.cos(th)+ey*np.sin(th)
        # eth = -ex*np.sin(th)+ey*np.cos(th)
        Rot = np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
        vcar = Rot@vrad
        return vcar
    
    def car2rad(self,vcar,th):
        # express vcar in polar coordinate
        # ex = er*np.cos(th)-eth*np.sin(th)
        # ey = er*np.sin(th)+eth*np.cos(th)
        Rot = np.array([[np.cos(th),np.sin(th)],[-np.sin(th),np.cos(th)]])
        vrad = Rot@vcar
        return vrad
        
        
    ##########################
    #### unused functions ####
    ##########################
    """
    def dynamicspolar2d(self,t,X,U,m1,w):
        # compute 2d version of chief dynamics
        r = X[0]
        v = X[2]
        om = X[3]
        u1 = U[0]
        u2 = U[1]
        mu = self.mu
        drhodv = v
        dphdt = om
        dvdt = r*om**2-mu/r**2+u1+w
        domdt = -2*v*om/r+u2/r+m1/r
        dXdt = np.array([drhodv,dphdt,dvdt,domdt])
        return dXdt
        
    def residualfilterXj(self,t,Zj2d,Yj,Uj,Xc,Uc,m1c,wc):
        # compute single dynamics of deputy FDI filter
        B = self.Bj2d
        C = self.Cj2d
        N = self.Nj2d
        nX2d = self.nX2d
        nY = self.nY
        nU2d = self.nU2d
        Xhatj2d = Zj2d+N@Yj
        phij = self.residualphij(t,Xhatj2d,Xc,Uc,m1c,wc)
        phifun = lambda X: self.residualphij(t,X,Xc,Uc,m1c,wc)
        phij = phifun(Xhatj2d)
        Aphij = self.getdfdX(Xhatj2d,phifun)
        I = np.identity(nX2d)
        M = I-N@C
        G = M@B
        Aekf = M@Aphij
        Q = np.identity(nX2d)*10
        R = np.identity(nY)
        KT,_,_ = control.lqr(Aekf.T,C.T,Q,R)
        K = -KT.T
        F = K@C
        H = K@C@N-K
        dZj2d_dt = F@Zj2d+G@Uj[0:nU2d]+M@phij+H@Yj
        return dZj2d_dt
    
    def residualfilterXcEKF(self,t,Zc,Yc,Uc):
        # compute dynamics of chief FDI filter with EKF
        nX2d = self.nX2d
        nY = self.nY
        nU2d = self.nU2d
        r_meas = Yc[0]
        B = np.array([[0,0],[1,0],[0,r_meas],[0,0]])
        Pw = np.array([[0],[0],[1],[0]])
        C = np.array([[1,0,0,0],[0,0,1,0],[0,0,0,1]])
        CP = C@Pw
        N = Pw@np.linalg.pinv((CP).T@CP)@CP.T
        self.Nekf = N
        Xhat_c = Zc+N@Yc
        phi = self.residualphiXc(Xhat_c)
        Aphi = self.getdfdX(Xhat_c,self.residualphiXc)
        I = np.identity(nX2d)
        M = I-N@C
        G = M@B
        Aekf = M@Aphi
        Q = np.identity(nX2d)*0.01
        R = np.identity(nY)
        KT,_,_ = control.lqr(Aekf.T,C.T,Q,R)
        K = -KT.T
        F = K@C
        H = K@C@N-K
        dZc_dt = F@Zc+G@Uc[0:nU2d]+M@phi+H@Yc
        return dZc_dt
    
    def residualphiXc(self,Xc2d):
        # compute unperturbed vector field of of chief FDI filter with EKF
        mu = self.mu
        r = Xc2d[0]
        vx = Xc2d[1]
        h = Xc2d[2]
        phi = np.array([vx,-mu/r**2+h**2/r**3,0,h/r**2])
        return phi
    
    def residualfilterXjall(self,t,Zj2dall,Yjall,Ujall,Xc,Uc,m1c,wc):
        # stack dynamics of deputy FDI filters
        p = self.Nsc
        dZj2d_dt_all = np.zeros_like(Zj2dall)
        nX2d = self.nX2d
        nY = self.nY
        for j in range(p):
            Zj2d = Zj2dall[nX2d*j:nX2d*(j+1)]
            Yj2d = Yjall[nY*j:nY*(j+1)]
            Uj2d = Ujall[nY*j:nY*(j+1)]
            dZj2d_dt = self.residualfilterXj(t,Zj2d,Yj2d,Uj2d,Xc,Uc,m1c,wc)
            dZj2d_dt_all[nX2d*j:nX2d*(j+1)] = dZj2d_dt
        return dZj2d_dt_all
    
    def measurementXj(self,t,Xj):
        # compute system output of single deputy S/C
        xj = Xj[0]
        yj = Xj[1]
        dyjdt = Xj[4]
        Yj = np.array([xj,yj,dyjdt])
        return Yj
    
    def measurementXjall(self,t,Xjall):
        # stack system outputs of deputy S/C
        p = self.Nsc
        nX = self.nX
        nY = self.nY
        Yjall = np.zeros(nY*p)
        for j in range(p):
            Xj = Xjall[nX*j:nX*(j+1)]
            Yj = self.measurementXj(t,Xj)
            Yjall[nY*j:nY*(j+1)] = Yj
        return Yjall
    
    def getXhatj2dall(self,Zj2dall,Yjall):
        # compute estimated state of deputy S/C
        p = self.Nsc
        N = self.Nj2d
        nX2d = self.nX2d
        nY = self.nY
        Xhatj2d_all = np.zeros_like(Zj2dall)
        for j in range(p):
            Zj2d = Zj2dall[nX2d*j:nX2d*(j+1)]
            Yj2d = Yjall[nY*j:nY*(j+1)]
            Xhatj2d = Zj2d+N@Yj2d
            Xhatj2d_all[nX2d*j:nX2d*(j+1)] = Xhatj2d
        return Xhatj2d_all
    
    def rad2car(self,vrad,th):
        # express vrad in cartesian coordinate
        Rot = np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
        vcar = Rot@vrad
        return vcar
    
    def getXdt(self,t):
        # compute target trajectory
        Qd_t = self.Qdfun(t)
        Tall = self.Tall 
        p = self.Nsc
        qd_val = Qd_t[0,:]
        qd_dot_val = Qd_t[1,:]
        qd_ddot_val = Qd_t[2,:]
        R_d = self.R_d
        qd_all = np.tile(qd_val,p)
        qd_dot_all = np.tile(qd_dot_val,p)
        qd_ddot_all = np.tile(qd_ddot_val,p)
        qd_all_trans = R_d.T@Tall@R_d@qd_all
        qd_dot_all_trans = R_d.T@Tall@R_d@qd_dot_all
        qd_ddot_all_trans = R_d@Tall@R_d@qd_ddot_all
        #qd_all_trans = np.linalg.solve(R_d,Tall@R_d@qd_all)
        #qd_dot_all_trans = np.linalg.solve(R_d,Tall@R_d@qd_dot_all)
        #qd_ddot_all_trans = np.linalg.solve(R_d,Tall@R_d@qd_ddot_all)
        Xd = self.Rqq2X@np.hstack((qd_all_trans,qd_dot_all_trans))
        return qd_all_trans,qd_dot_all_trans,qd_ddot_all_trans,Xd
    """
    
    
if __name__ == "__main__":
    # simulation parameters
    Nsc = 4 # number of S/C
    mu_earth = 3.9860e+05 # gravitational constant of Earth
    mu_mars = 42828.375816 # gravitational constant of Mars 
    mu = mu_mars
    bm = Benchmark(mu,Nsc)
    t0 = 0
    
    # chief S/C initial condition
    r0_earth = 6378 # radius of Earth
    r0_mars = 3396 # radius of mars
    r0 = r0_mars+1000
    vx0 = 0
    vth0 = np.sqrt(bm.mu/r0)
    om0 = vth0/r0
    h0 = om0*r0**2 
    Om0 = np.pi/3
    i0 = np.pi/3
    th0 = 0
    Xc0 = np.array([r0,vx0,h0,Om0,i0,th0])
    
    # soft landing target trajectory
    omar = 870/(60*60)/r0_mars # rotation speed of Mars
    hmar = r0_mars**2*omar
    X0sl = np.array([r0,vx0,h0,np.pi/2])
    Xfsl = np.array([r0_mars,0,hmar,0])
    Xs = bm.softlandingscp(X0sl,Xfsl)
    
    # deputy S/C initial conditions
    Xjcall0 = np.hstack((bm.Xjall0,Xc0))
    xi_c0 = h0
    Yjall0 = bm.measurementnet(t0,bm.Xjall0)
    Zj2dall0 = np.zeros(4*Nsc)
    for j in range(Nsc):
        Xj0 = bm.Xjall0[6*j:6*(j+1)]
        Yj0 = Yjall0[3*j:3*(j+1)]
        Xhatj2d0 = np.array([Xj0[0],Xj0[1],Xj0[3],Xj0[4]])
        Zj2dall0[4*j:4*(j+1)] = Xhatj2d0-bm.Nj2d@Yj0
    
    # target trajectory for relative dynamics
    T = 2*np.pi/om0/10
    dt = bm.h
    Nint = int(T/dt)
    Qd_t_his = np.zeros((3,3,Nint+1))
    ti = 0
    for i in range(Nint+1):
        Qd_t_his[:,:,i] = bm.Qdfun(ti)
        ti = ti+dt
    
    # initialization
    this = np.zeros(Nint+1)
    Xjcall_his = np.zeros((Nint+1,6*(Nsc+1)))
    Zj2dall_his = np.zeros((Nint+1,4*Nsc))
    res_jc_his = np.zeros((Nsc+1,Nint))
    this[0] = t0
    Xjcall_his[0,:] = Xjcall0
    Zj2dall_his[0,:] = Zj2dall0
    t = t0
    Xjcall = Xjcall0
    xi_c = xi_c0
    Zj2dall = Zj2dall0
    
    # multi-S/C fault detection and isolation
    Nfaults = 10
    Nswitches = 100
    dist_period = 300
    fsize = 0.3
    m1jc_his,wjc_his = bm.generatesignals(Nint,Nfaults,dist_period,fsize)
    array_switches = np.arange(1,Nint-1)
    idx_switches = np.random.choice(array_switches,Nswitches,False)
    idx_switches = set(idx_switches)
    
    # integration
    for k in range(Nint):
        if k in idx_switches:
            bm.changecommunication()
        Uc = np.zeros(3)
        Xjall = Xjcall[0:6*Nsc]
        Xj2dall = bm.R3d22d@Xjall
        Xc = Xjcall[6*Nsc:6*(Nsc+1)]
        Qd_t = Qd_t_his[:,:,k]
        Ujall = bm.expcontrol(t,Xjall,Xc,Uc,Qd_t)
        Ujcall = np.hstack((Ujall,Uc))
        Yc = bm.measurementXc(t,Xc)
        #Yjall = bm.measurementXjall(t,Xjall)
        #Xhatj2d_all = bm.getXhatj2dall(Zj2dall,Yjall)
        Yjall = bm.measurementnet(t,Xjall)
        Xhatj2d_all = Zj2dall+bm.N2dall@bm.Ctrans@Yjall
        for j in range(Nsc):
            Xj2d = Xj2dall[4*j:4*(j+1)]
            Xhatj2d = Xhatj2d_all[4*j:4*(j+1)]
            res_jc_his[j,k] = np.linalg.norm(Xhatj2d-Xj2d)
            res_jc_his[Nsc,k] = np.linalg.norm(Yc[1]-xi_c)
        m1jcall = m1jc_his[:,k]
        wjcall = wjc_his[:,k]
        dynamics_jc = lambda t,X,U: bm.dynamicsall(t,X,U,m1jcall,wjcall)
        dynamics_rc = lambda t,X,U: bm.residualfilterXc(t,X,Yc,U)
        #dynamics_rj = lambda t,X,U: bm.residualfilterXjall(t,X,Yjall,U,Xc,Uc,m1jcall[Nsc],wjcall[Nsc])
        dynamics_rj = lambda t,X,U: bm.residualfilternet(t,X,Yjall,U,Xc,Uc,m1jcall[Nsc],wjcall[Nsc])
        t,Xjcall = bm.rk4(t,Xjcall,Ujcall,dynamics_jc)
        _,xi_c = bm.rk4(t,xi_c,Uc,dynamics_rc)
        _,Zj2dall = bm.rk4(t,Zj2dall,Ujall,dynamics_rj)
        this[k+1] = t
        Xjcall_his[k+1,:] = Xjcall
        Zj2dall_his[k+1,:] = Zj2dall
    
    # figures
    Xchis = Xjcall_his[:,6*Nsc:6*(Nsc+1)]
    Xjallhis = Xjcall_his[:,0:6*Nsc]
    x3dhis = Xchis[:,0]*np.cos(Xchis[:,5])
    y3dhis = Xchis[:,0]*np.sin(Xchis[:,5])
    
    # chief residual filter
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(this[0:Nint],res_jc_his[Nsc,:],"C0")
    ax2.plot(this[0:Nint],m1jc_his[Nsc,:],"C1",linestyle="--")
    ax2.plot(this[0:Nint],wjc_his[Nsc,:],"C2",linestyle="--")
    ax2.legend(["fault signal","disturbance signal"])
    ax1.set_xlabel("time")
    ax1.set_ylabel("residual")
    ax2.set_ylabel("fault and disturbance signal")
    
    # deputy residual filter
    for j in range(Nsc):
        fig,ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(this[0:Nint],res_jc_his[j,:],"C0")
        ax2.plot(this[0:Nint],m1jc_his[j,:],"C1",linestyle="--")
        ax2.plot(this[0:Nint],wjc_his[j,:],"C2",linestyle="--")
        ax2.legend(["fault signal","disturbance signal"])
        ax1.set_xlabel("time")
        ax1.set_ylabel("residual")
        ax2.set_ylabel("fault and disturbance signal")
        
    # relative positions
    plt.figure()
    plt.plot(x3dhis,y3dhis,linestyle=':')
    plt.figure()
    for j in range(Nsc):
        Xjhis = Xjallhis[:,6*j:6*(j+1)]
        plt.plot(Xjhis[:,0],Xjhis[:,1])