"""Cartesian HHT inertia projected before runtime, with no volume traversal.

For a fixed unit rotation axis C^3=-C. All rotated inertial products therefore
use three precomputed Gram matrices of [X-center, Phi]. The formula matches
the Cartesian full-coordinate diagnostic on the retained displacement span.
Residual surface coordinates remain quasistatic; material history is separate.
"""
import numpy as np
import scipy.linalg as la


def prepare_inertia(FE,principal,body):
    if not body.revolute:return None
    ev,Q=la.eigh(principal.T@(FE.K@principal),principal.T@(FE.M@principal))
    P=principal@Q;V=np.zeros((FE.X.size,P.shape[1]));V[FE.free]=P
    if not np.allclose(V[body.original_surface_rows],body.P,rtol=1e-8,atol=1e-10):
        raise ValueError('Offline modal orientation differs from runtime basis')
    y=(FE.X-FE.center).ravel();free=np.zeros(FE.X.size,dtype=bool);free[FE.free]=True
    mass=FE.mass*free;F=np.column_stack((y,V))
    cf=np.cross(FE.axis,F.reshape(-1,3,F.shape[1]).transpose(0,2,1)).transpose(0,2,1).reshape(F.shape)
    c2f=np.cross(FE.axis,cf.reshape(-1,3,F.shape[1]).transpose(0,2,1)).transpose(0,2,1).reshape(F.shape)
    H=[F.T@(mass[:,None]*z) for z in (F,cf,c2f)]
    total_I=body.I0;free_I=-H[2][0,0]
    assert abs(H[0][1:,1:]-body.M).max(initial=0.)<1e-8
    data=dict(H0=H[0],H1=H[1],H2=H[2],Minv=la.inv(body.M) if body.M.size else np.empty((0,0)),hub_I=total_I-free_I)
    body.cartesian_inertia=data
    return data


class ProjectedInertia:
    def __init__(self,data,xp=np):
        self.xp=xp
        for k,v in data.items():setattr(self,k,xp.asarray(v))
        if xp is not np:
            self.mv_kernel=xp.RawKernel(r'''
extern "C" __global__ void projected_mv(const double*A,const double*x,double*y,int rows,int cols,long long rs,long long cs,long long xs){
 int r=blockIdx.x,t=threadIdx.x;double v=0.;
 for(int j=t;j<cols;j+=128)v+=A[r*rs+j*cs]*x[j*xs];
 __shared__ double sums[128];sums[t]=v;__syncthreads();
 for(int s=64;s;s/=2){if(t<s)sums[t]+=sums[t+s];__syncthreads();}if(!t)y[r]=sums[0];
}
''','projected_mv')
    def mv(self,A,x):
        if self.xp is np:return A@x
        y=self.xp.empty(A.shape[0])
        if len(y):self.mv_kernel((len(y),),(128,),(A,x,y,np.int32(A.shape[0]),np.int32(A.shape[1]),
            np.int64(A.strides[0]//8),np.int64(A.strides[1]//8),np.int64(x.strides[0]//8)))
        return y
    def homogeneous(self,q):return self.xp.concatenate((self.xp.ones(1),q[1:]))
    def elastic(self,q):return self.xp.concatenate((self.xp.zeros(1),q[1:]))
    def predictor(self,q,v,a,h,beta,gamma):
        y=self.homogeneous(q);dv=self.elastic(v);da=self.elastic(a);w=v[0];al=a[0];b=h*h*(.5-beta);c=h*(1-gamma)
        pos=(y+h*dv+b*da,h*w*y+b*(2*w*dv+al*y),b*w*w*y)
        vel=(dv+c*da,w*y+c*(2*w*dv+al*y),c*w*w*y)
        return q[0].copy(),pos,vel
    def rotate(self,z,angle):
        s=self.xp.sin(angle);c=self.xp.cos(angle);a,b,d=z
        return a,c*b+s*(a-d),c*d+(1-c)*a+s*b
    def modal_projection(self,z):
        return (self.mv(self.H0,z[0])+self.mv(self.H1,z[1])+self.mv(self.H2,z[2]))[1:]
    def acceleration(self,q,predictor,h,beta):
        angle,pos,_=predictor;p=self.rotate(pos,angle-q[0]);c=1/(beta*h*h)
        return ((self.homogeneous(q)-p[0])*c,-p[1]*c,-p[2]*c)
    def force(self,q,acc,predictor,h,beta):
        z=self.acceleration(q,predictor,h,beta);y=self.homogeneous(q)
        torque=self.xp.sum(y*(-self.mv(self.H1,z[0])-self.mv(self.H2,z[1])+self.mv(self.H1,z[2])))+self.hub_I*acc[0]
        return self.xp.concatenate((torque.reshape(1),self.modal_projection(z)))
    def velocities(self,q,v,a,predictor,h,beta,gamma):
        angle,_,vb=predictor;vb=self.rotate(vb,angle-q[0]);z=self.acceleration(q,predictor,h,beta)
        y=self.homogeneous(q);w=v[0];al=a[0];cy=self.mv(self.H1,y)[1:];c2y=self.mv(self.H2,y)[1:]
        vmodal=self.mv(self.Minv,self.modal_projection(vb)+gamma*h*self.modal_projection(z)-w*cy)
        amodal=self.mv(self.Minv,self.modal_projection(z)-2*w*self.mv(self.H1[1:,1:],vmodal)-al*cy-w*w*c2y)
        return vmodal,amodal
    def zero_physical_acceleration(self,q,v):
        return self.mv(self.Minv,-v[0]*v[0]*self.mv(self.H2,self.homogeneous(q))[1:])


def cpu_class():
    from residual_cms import ResidualSurfaceCMS
    from surface_cms import inertia
    class ProjectedCartesianCMS(ResidualSurfaceCMS):
        def __init__(self,*args,**kw):
            super().__init__(*args,**kw)
            self.frame_ops=[ProjectedInertia(b.cartesian_inertia) if b.revolute else None for b in self.bodies]
            for i,op in enumerate(self.frame_ops):
                if op is not None:
                    lo,hi=self.doff[i:i+2];self.acc[lo+1:hi]=op.zero_physical_acceleration(self.d[lo:hi],self.v[lo:hi])
        def nonmass(self,d,v,t):
            r=super().nonmass(d,v,t)
            for i,b in enumerate(self.bodies):
                if b.revolute:
                    lo,hi=self.doff[i:i+2];r[lo:hi]-=inertia(b,d[lo:hi],v[lo:hi])[3]
            return r
        def evaluate(self,d,s,t,*args):
            old=[]
            for i,op in enumerate(self.frame_ops):
                if op is None:continue
                lo,hi=self.doff[i:i+2];block=self.blocks[i];old.append((block,block.mass_action))
                block.mass_action=lambda acc,op=op,q=d[lo:hi],p=self.frame_predictors[i]:op.force(q,acc,p,self.dt,self.beta)
            try:return super().evaluate(d,s,t,*args)
            finally:
                for b,method in old:b.mass_action=method
        def step(self):
            import time
            tick=time.perf_counter();h=self.dt
            self.frame_predictors=[op.predictor(self.d[lo:hi],self.v[lo:hi],self.acc[lo:hi],h,self.beta,self.gamma) if op else None
                for op,lo,hi in zip(self.frame_ops,self.doff[:-1],self.doff[1:])]
            row=super().step()
            for i,op in enumerate(self.frame_ops):
                if op is not None:
                    lo,hi=self.doff[i:i+2]
                    v,a=op.velocities(self.d[lo:hi],self.v[lo:hi],self.acc[lo:hi],self.frame_predictors[i],h,self.beta,self.gamma)
                    self.v[lo+1:hi]=v;self.acc[lo+1:hi]=a
            row['online_s']=time.perf_counter()-tick;return row
    return ProjectedCartesianCMS


def gpu_class(base):
    import cupy as cp
    from parallel_modal import ParallelModalKernels
    class CartesianModal(ParallelModalKernels):
        def evaluate(self,sim,d,v,acc,old,clock):
            nm,g,scale,free=super().evaluate(sim,d,v,acc,old,clock)
            for i,(b,B,op) in enumerate(zip(sim.bodies,sim.dynamic,sim.frame_ops)):
                if op is None:continue
                lo,hi=sim.doff[i:i+2];nm[lo:hi]-=B.bias
                g[lo:hi]=sim.w*nm[lo:hi]-sim.alpha*old[lo:hi]+op.force(d[lo:hi],acc[lo:hi],sim.frame_predictors[i],sim.dt,sim.beta)
                y=op.mv(B.inverse,g[lo+b.offset:hi]);theta=(g[lo]-cp.sum(B.h*y))/B.sigma
                free[lo]=-theta;free[lo+b.offset:hi]=-y+B.u*theta
            return nm,g,scale,free
    class ProjectedCartesianGPU(base):
        def __init__(self,scene,bodies,*args,**kw):
            self.frame_ops=[ProjectedInertia(b.cartesian_inertia,cp) if b.revolute else None for b in bodies]
            self.cartesian_initial_applied=False
            super().__init__(scene,bodies,*args,**kw)
        def nonmass(self,d,v,t):
            r=super().nonmass(d,v,t)
            for i,(b,B) in enumerate(zip(self.bodies,self.dynamic)):
                if b.revolute:
                    lo,hi=self.doff[i:i+2];r[lo:hi]-=B.bias
            return r
        def initialize(self):
            if not self.cartesian_initial_applied:
                for op,lo,hi in zip(self.frame_ops,self.doff[:-1],self.doff[1:]):
                    if op:self.acc[lo+1:hi]=op.zero_physical_acceleration(self.d[lo:hi],self.v[lo:hi])
                self.cartesian_initial_applied=True
            super().initialize()
            self.frame_predictors=[op.predictor(self.d[lo:hi],self.v[lo:hi],self.acc[lo:hi],self.dt,self.beta,self.gamma) if op else None
                for op,lo,hi in zip(self.frame_ops,self.doff[:-1],self.doff[1:])]
        def rebuild(self,refresh=True):
            if not isinstance(self.modal_kernels,CartesianModal):self.modal_kernels=CartesianModal()
            return super().rebuild(refresh)
        def finish(self):
            for i,op in enumerate(self.frame_ops):
                if op is not None:
                    lo,hi=self.doff[i:i+2]
                    v,a=op.velocities(self.td[lo:hi],self.tv[lo:hi],self.ta[lo:hi],self.frame_predictors[i],self.dt,self.beta,self.gamma)
                    self.tv[lo+1:hi]=v;self.ta[lo+1:hi]=a
            return super().finish()
    return ProjectedCartesianGPU
