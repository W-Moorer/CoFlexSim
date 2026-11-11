"""CUDA numerical path for the frozen one-axis Surface-CMS/HHT model.

All dynamical states, geometry, operators, forces and solves live on CUDA.
Python dispatches kernels and reads scalar stopping/pivot decisions. This is not
yet a persistent device scheduler or a general six-DOF floating-frame model.
"""
import time
from types import SimpleNamespace
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as csp
from device_geometry import DeviceGeometry
from device_qp import working_set,small_lcp
from operators import SparseColumnKernel
from csr_actions import ContactActions
from modal_step import ModalKernels
from working_graph import GraphWorkingSet


HELPERS=r'''
extern "C" __global__ void rotate_rows(const double*in,double*out,int n,const double*R){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=n)return;
 for(int a=0;a<3;a++)out[3*j+a]=in[3*j]*R[a]+in[3*j+1]*R[3+a]+in[3*j+2]*R[6+a];
}
extern "C" __global__ void sparse_load_response(const double*R,const double*f,const long long*ids,double*out,int n,int k){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;double value=0.;
 for(int j=0;j<k;j++){long long c=ids[j];value+=R[c*n+i]*f[c];}out[i]=value;
}
'''


def sync():cp.cuda.get_current_stream().synchronize()


class DeviceBlock:
    def __init__(self,b,factor,stiffness_weight=1.):
        self.source=b;self.factor=factor;self.offset=b.offset;self.reference=b.reference;self.stiffness_weight=stiffness_weight
        for k in ('P','K','M','Q','D','c0','d0','mt','X','axis','center'):
            setattr(self,k,cp.ascontiguousarray(cp.asarray(getattr(b,k))))
        self.base=stiffness_weight*self.K+factor*self.M;self.inverse=cp.ascontiguousarray(cp.linalg.inv(self.base))
        self.initialize_surface_response(b)
        a=b.axis;self.cross=cp.asarray([[0.,-a[2],a[1]],[a[2],0.,-a[0]],[-a[1],a[0],0.]])
        self.cross2=self.cross@self.cross
        n=self.K.shape[0];self.c=cp.zeros(n);self.h=cp.zeros(n);self.u=cp.zeros(n)
        self.I=cp.asarray(1.);self.sigma=cp.asarray(1.);self.bias=cp.zeros(b.rd)

    def initialize_surface_response(self,b):
        self.R=cp.asfortranarray(cp.asarray(b.R)) if b.R is not None else None
        self.C=self.P@self.inverse@self.P.T
        if self.R is not None:self.C+=self.R
        self.C=cp.asfortranarray(self.C)

    def update(self,q,v,guide):
        b=self.source;a=q[b.offset:];ad=v[b.offset:];self.bias=cp.zeros(b.rd);self.guide=guide
        if not b.reference:return
        if b.prismatic:
            self.c=self.mt;self.I=cp.asarray(b.total_mass)
        else:
            w=v[0];Da=self.D@a;self.c=self.c0+self.Q@a
            self.I=b.I0-2*self.d0@a-a@Da
            self.bias[1:]=2*w*(self.Q@ad)+w*w*(self.d0+Da)
            self.bias[0]=-2*w*((self.d0+Da)@ad)
        self.h=self.factor*self.c;self.u=self.inverse@self.h
        self.sigma=self.factor*self.I+guide-self.h@self.u

    def solve(self,r):
        y=self.inverse@r[self.offset:]
        if not self.reference:return y
        theta=(r[0]-self.h@y)/self.sigma
        return cp.concatenate((theta.reshape(1),y-self.u*theta))

    def mass(self,a):
        if not self.reference:return self.M@a
        return cp.concatenate(((self.I*a[0]+self.c@a[1:]).reshape(1),self.c*a[0]+self.M@a[1:]))

    def action(self,a):
        y=self.factor*self.mass(a);y[self.offset:]+=self.stiffness_weight*(self.K@a[self.offset:])
        if self.reference:y[0]+=self.guide*a[0]
        return y

    def diagonal(self):
        v=cp.diag(self.base)
        return cp.concatenate(((self.factor*self.I+self.guide).reshape(1),v)) if self.reference else v


class DeviceMapping:
    def __init__(self,sim,d,s):
        self.sim=sim;self.parts=[];self.x=cp.empty((sim.ns//3,3))
        for i,(b,block) in enumerate(zip(sim.bodies,sim.dynamic)):
            dl,dh=sim.doff[i:i+2];sl,sh=sim.soff[i:i+2];q=d[dl:dh]
            self.parts.append(sim.modal_kernels.map_body(block,q,s[sl:sh],self.x[sl//3:sh//3]))

    def apply(self,direction):
        sim=self.sim;out=cp.empty(sim.ns)
        for i,(b,block,(R,tangent)) in enumerate(zip(sim.bodies,sim.dynamic,self.parts)):
            dl,dh=sim.doff[i:i+2];sl,sh=sim.soff[i:i+2]
            sim.modal_kernels.apply_body(block,direction[dl:dh],direction[sim.nd+sl:sim.nd+sh],R,tangent,out[sl:sh])
        return out


class DeviceSurfaceCMS:
    def __init__(self,scene,bodies):
        start=time.perf_counter();self.bodies=bodies;self.scene=scene;self.dt=float(scene['dt']);self.cp=cp;self.dtype=cp.dtype('float64');self.column_major=True
        self.alpha=-.05;self.w=1+self.alpha;self.beta=(1-self.alpha)**2/4;self.gamma=.5-self.alpha
        self.endpoint_contact=scene.get('contact_force_convention','weighted')=='physical_endpoint'
        self.policy=dict(contact_qp_tolerance=1e-14,relative_tolerance=1e-6,absolute_tolerance=1e-9,
                         gap_tolerance=1e-9,static_tolerance_m=1e-11,max_iterations=150)
        self.doff=np.r_[0,np.cumsum([b.rd for b in bodies])];self.nd=int(self.doff[-1])
        self.soff=np.r_[0,np.cumsum([b.X.size for b in bodies])];self.ns=int(self.soff[-1]);self.x_offsets=self.soff//3
        block_type=getattr(self,'block_type',DeviceBlock)
        self.dynamic=[block_type(b,1/(self.beta*self.dt**2*(1. if self.endpoint_contact else self.w)),
            self.w if self.endpoint_contact else 1.) for b in bodies]
        self.geometry=DeviceGeometry(bodies,scene,use_graph=True);self.actions=ContactActions();self.modal_kernels=ModalKernels();self.helpers=cp.RawModule(code=HELPERS)
        self.rotate_kernel=self.helpers.get_function('rotate_rows');self.recover_kernel=self.helpers.get_function('sparse_load_response')
        self.d=cp.zeros(self.nd);self.v=cp.zeros(self.nd);self.acc=cp.zeros(self.nd);self.s=cp.zeros(self.ns);self.p=cp.zeros(self.ns)
        for i,b in enumerate(bodies):
            if b.reference:
                self.d[self.doff[i]]=b.spec.get('initial_angle' if b.revolute else 'initial_reference_position',0.)
                self.v[self.doff[i]]=b.spec.get('initial_angular_velocity' if b.revolute else 'initial_reference_velocity',0.)
        self.history=[];self.t=0.;self.x=DeviceMapping(self,self.d,self.s).x
        self.statistics=dict(online_volume_element_evaluations=0,online_volume_reconstructions=0,
                             online_full_order_factorizations=0,contact_restricted_solves=0,device_response_columns=0)
        self.graph_working=GraphWorkingSet(self)
        self.old_nonmass=self.nonmass(self.d,self.v,0.)
        # Compile and initialize numerical kernels before the measured trajectory.
        self.geometry.build(self.x)
        small_lcp(cp.eye(2),cp.asarray([-1.,1.]))
        sync();self.precompute_s=time.perf_counter()-start

    def nonmass(self,d,v,t):
        out=cp.zeros(self.nd);ramp=min(1.,t/max(self.dt,self.scene.get('load_ramp',self.dt)))
        for i,(b,block) in enumerate(zip(self.bodies,self.dynamic)):
            lo,hi=self.doff[i:i+2];q=d[lo:hi];speed=v[lo:hi]
            k=b.spec.get('guide_stiffness',0.);c=b.spec.get('guide_damping',0.);target=b.spec.get('guide_target_velocity',0.)
            guide=(k+c*self.gamma/(self.beta*self.dt))*(self.w if self.endpoint_contact else 1.);block.update(q,speed,guide)
            out[lo:hi]=block.bias;out[lo+b.offset:hi]+=block.K@q[b.offset:]
            if b.reference:
                out[lo]-=b.spec.get('torque' if b.revolute else 'reference_force',0.)*ramp
                out[lo]+=k*(q[0]-b.spec.get('guide_rest_coordinate',0.)-target*t)+c*(speed[0]-target)
        return out

    def prepare_contact(self,world,mapping):
        self.n=len(world.gaps);self.blocks=[];self.contact=[]
        for i,(b,block,(R,tangent)) in enumerate(zip(self.bodies,self.dynamic,mapping.parts)):
            sl,sh=self.soff[i:i+2]
            pu=block.P@block.u if b.reference else cp.empty(0)
            B,jr,v,rank=self.actions.body(world.G,R,tangent,pu,block.sigma if b.reference else cp.asarray(1.),int(sl),int(sh-sl))
            self.blocks.append((B,block.C,rank));self.contact.append((B,jr,v))

    def columns(self,ids):
        self.statistics['device_response_columns']+=len(ids)
        return SparseColumnKernel(self,ids,capture=False,warmup=False).compute()

    def contact_response(self,lam,free):
        direction=free.copy();targets=cp.zeros(self.ns);forces=cp.zeros(self.ns);targetd=cp.zeros(self.nd)
        for i,(b,block,(B,jr,v)) in enumerate(zip(self.bodies,self.dynamic,self.contact)):
            dl,dh=self.doff[i:i+2];sl,sh=self.soff[i:i+2];f=self.actions.adjoint(B,lam);fm=block.P.T@f
            forces[sl:sh]=f;targetd[dl+b.offset:dh]=fm;y=block.inverse@fm
            if b.reference:
                theta=(v@lam)/block.sigma;direction[dl]+=theta;y-=block.u*theta;targetd[dl]=jr@lam
            direction[dl+b.offset:dh]+=y
            if block.R is not None:
                ids=cp.flatnonzero(f).astype(cp.int64);n=sh-sl
                self.recover_kernel(((n+127)//128,),(128,),(block.R,f,ids,targets[sl:sh],np.int32(n),np.int32(len(ids))))
        return direction,targets,forces,targetd

    def step(self):
        sync();tick=time.perf_counter();h=self.dt;t=self.t+h
        predictor=self.d+h*self.v+h*h*(.5-self.beta)*self.acc;velocity_base=self.v+h*(1-self.gamma)*self.acc
        d=predictor.copy();s=self.s.copy();p=self.p.copy();reference=None
        timers=dict(geometry=0.,coupled_solve=0.,surface_mapping=0.,modal=0.)
        for iteration in range(self.policy['max_iterations']):
            start=time.perf_counter();acc=(d-predictor)/(self.beta*h*h);v=velocity_base+self.gamma*h*acc
            nm,g,scale,free=self.modal_kernels.evaluate(self,d,v,acc,self.old_nonmass,t)
            if any(b.reference and float(B.sigma)<=0. for b,B in zip(self.bodies,self.dynamic)):
                raise ValueError('Nonpositive floating-frame inertia Schur complement')
            sync();timers['modal']+=time.perf_counter()-start
            start=time.perf_counter();mapping=DeviceMapping(self,d,s);x=mapping.x;sync();timers['surface_mapping']+=time.perf_counter()-start
            start=time.perf_counter();world=self.geometry.build(x);gaps=world.gaps;sync();timers['geometry']+=time.perf_counter()-start
            start=time.perf_counter()
            free[self.nd:]=-s
            b=gaps+self.actions.action(world.G,mapping.apply(free))
            if len(gaps) and float(cp.min(b))<0.:
                self.prepare_contact(world,mapping)
                lam,info=self.graph_working.solve(b,self.geometry.seeds(world,b),self.policy['contact_qp_tolerance'])
                direction,targets,targetp,targetd=self.contact_response(lam,free)
                self.statistics['contact_restricted_solves']+=info['outer_iterations']
            else:
                lam=cp.zeros(len(gaps));direction=free;targets=cp.zeros(self.ns);targetp=targets.copy();targetd=cp.zeros(self.nd)
            direction[self.nd:]=targets-s
            residual=float(cp.linalg.norm((g-targetd)/scale));static_error=float(cp.max(cp.abs(targets-s)))
            if reference is None:reference=max(float(cp.linalg.norm(g/scale)),1e-12)
            threshold=self.policy['absolute_tolerance']+self.policy['relative_tolerance']*reference
            pen=max(0.,-float(cp.min(gaps))) if len(gaps) else 0.;comp=float(cp.max(cp.abs(lam*gaps))) if len(gaps) else 0.
            timers['coupled_solve']+=time.perf_counter()-start
            if residual<=threshold and static_error<=self.policy['static_tolerance_m'] and pen<=self.policy['gap_tolerance'] and comp<=1e-8:break
            dd=direction[:self.nd];ds=direction[self.nd:];move=float(cp.max(cp.linalg.norm(mapping.apply(direction).reshape(-1,3),axis=1)))
            step=min(1.,.4*self.geometry.radius/max(move,1e-20))
            for i,bod in enumerate(self.bodies):
                if bod.revolute:step=min(step,.025/max(abs(float(dd[self.doff[i]])),1e-20))
            penalty=cp.maximum(1.,1.5*cp.sum(lam));base_static=.5*p@s;slope=g@dd+p@ds-penalty*pen
            curvature=sum(dd[self.doff[i]:self.doff[i+1]]@block.action(dd[self.doff[i]:self.doff[i+1]]) for i,block in enumerate(self.dynamic))
            accepted=False;start=time.perf_counter()
            for backtrack in range(30):
                dt=d+step*dd;st=s+step*ds;pt=p+step*(targetp-p);xt=DeviceMapping(self,dt,st).x
                gt=self.geometry.build(xt,derivatives=False);vp=cp.maximum(0.,-cp.min(gt)) if len(gt) else cp.asarray(0.)
                change=step*(g@dd)+.5*step*step*curvature+.5*pt@st-base_static+penalty*(vp-pen)
                if bool(change<=1e-4*step*cp.minimum(slope,0.)+1e-15):
                    d,s,p=dt,st,pt;accepted=True;break
                step*=.5
            timers['geometry']+=time.perf_counter()-start
            if not accepted:raise RuntimeError(f'GPU line search failed: {iteration}, {residual=}, {static_error=}, {pen=}')
        else:raise RuntimeError(f'GPU nonlinear limit: {residual=}, {static_error=}, {pen=}')
        self.d=d;self.s=s;self.p=p;self.acc=acc;self.v=v;self.t=t;self.x=x;self.old_nonmass=nm if self.endpoint_contact else nm-targetd
        self.geometry.commit(world,lam);nodal=self.actions.adjoint(world.G,lam).reshape(-1,3)
        self.statistics.update(graph_contact_calls=self.graph_working.calls,graph_contact_fallbacks=self.graph_working.fallbacks,
                               graph_contact_rebuilds=self.graph_working.rebuilds)
        force=cp.stack([cp.sum(nodal[self.x_offsets[i]:self.x_offsets[i+1]],axis=0) for i in range(len(self.bodies))])
        row=dict(t=t,iterations=iteration+1,residual=residual,threshold=threshold,static_equilibrium_error_m=static_error,
                 complementarity_Nm=comp,pen_um=max(0.,-float(world.physical_min))*1e6,
                 force=cp.asnumpy(force).tolist(),refs=[float(d[self.doff[i]]) for i,bod in enumerate(self.bodies) if bod.reference],timers=timers)
        sync();row['online_s']=time.perf_counter()-tick;self.last_nodal_force=nodal;self.history.append(row);return row
