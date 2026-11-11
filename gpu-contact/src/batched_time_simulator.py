"""GPU time-step loop with complete compact histories and explicit failures."""
import time
import numpy as np
import cupy as cp
from parallel_loop_simulator import ParallelLoopSurfaceCMS
from conditional_graph import NativeGraph
from trajectory_graph import trajectory_graph
from device_simulator import sync

CODE=r'''
extern "C" __global__ void record_time_step(const int*outer,int*run,double*clock,double h,
 const double*report,double*history,int width,int capacity){
 __shared__ int row,ok;
 if(!threadIdx.x){row=run[1];ok=outer[4]&&!outer[5]&&row<capacity;
  if(!ok)run[2]=outer[5]?outer[5]:98;}
 __syncthreads();if(!ok)return;
 for(int i=threadIdx.x;i<width;i+=blockDim.x)history[row*(width+1)+1+i]=report[i];
 if(!threadIdx.x)history[row*(width+1)]=clock[0];
 __syncthreads();if(!threadIdx.x){run[1]++;clock[0]+=h;}
}
'''

class BatchedTimeSurfaceCMS(ParallelLoopSurfaceCMS):
    def __init__(self,scene,bodies,policy=None,batch_capacity=256):
        self.batch_capacity=int(batch_capacity)
        if not 1<=self.batch_capacity<=4096:raise ValueError('Batch capacity must be 1..4096')
        self.run_state=cp.zeros(3,dtype=cp.int32);self.record_time=cp.RawKernel(CODE,'record_time_step')
        self.batch_launches=0;self.completed_rows=[];self.batch_history=[];self.trajectory=None
        super().__init__(scene,bodies,policy)

    def finish(self):
        super().finish()
        # Keep every accepted-step moment, too; batch size never changes outputs.
        self.base_width=int(self.output.size)
        moments=[cp.cross(self.x[lo:hi]-B.center,self.nodal[lo:hi]).sum(0)
            for B,lo,hi in zip(self.dynamic,self.x_offsets[:-1],self.x_offsets[1:])]
        self.output=cp.concatenate((self.output,cp.stack(moments).ravel()))

    def rebuild(self,refresh=True):
        super().rebuild(refresh)
        self.trajectory_width=int(self.output.size)+1
        self.device_history=cp.empty((self.batch_capacity,self.trajectory_width))
        def record():
            self.record_time((1,),(256,),(self.outer,self.run_state,self.clock,np.float64(self.dt),self.output,self.device_history,
                np.int32(self.output.size),np.int32(self.batch_capacity)))
        # v5's rebuild leaves outer.error=99 during graph construction.
        part=NativeGraph.capture(record)
        self.trajectory=trajectory_graph([*self.loop.refs[:5],part],self.outer,self.graph_working.meta,self.run_state)
        self.run_state.fill(0);sync()

    def unpack(self,data):
        rows=[];nf=3*len(self.bodies);rs=24+nf;re=rs+len(self.reference_ids)
        for record in data:
            physical=record[0];out=record[1:];state=out[:16].astype(int);vals=out[16:23]
            rows.append(dict(t=float(physical),iterations=int(state[1]),residual=float(vals[0]),threshold=float(vals[1]),
                static_equilibrium_error_m=float(vals[2]),complementarity_Nm=float(vals[4]),pen_um=float(max(0.,-out[23])*1e6),
                force=out[24:rs].reshape(-1,3).tolist(),refs=out[rs:re].tolist(),nonlinear_trial_evaluations=int(state[3]),
                device_evaluations=int(state[6]),contact_moment_about_reference_Nm=out[self.base_width:].reshape(-1,3).tolist()))
        return rows

    def advance_batch(self,steps):
        if not 1<=steps<=self.batch_capacity:raise ValueError('Invalid batch size')
        sync();start=time.perf_counter();accepted=[];completed=0
        for retry in range(8):
            self.run_state.set(np.asarray([steps-completed,0,0],dtype=np.int32));self.clock.fill(self.t+self.dt)
            self.trajectory.launch();self.batch_launches+=1
            state=cp.asnumpy(self.run_state);done=int(state[1]);error=int(state[2])
            data=cp.asnumpy(self.device_history[:done]);accepted.extend(self.unpack(data));completed+=done
            if done:self.t=float(data[-1,0])
            if not error:break
            self.exceptions+=1
            if error==16:
                if retry:self.geometry.skin*=2
                self.geometry.refresh(self.x);self.rebuild(refresh=False)
            elif error==105 and self.graph_working.cap<256:
                old=self.graph_working.cap;self.graph_working.cap*=2;self.graph_working.graph=None
                self.graph_working.oldkeys=cp.pad(self.graph_working.oldkeys,(0,old));self.graph_working.oldw=cp.pad(self.graph_working.oldw,(0,old));self.rebuild()
            else:
                self.completed_rows.extend(accepted)
                raise RuntimeError(f'Time loop stopped after {completed} accepted steps in batch, error {error}, time {self.t}')
        else:
            self.completed_rows.extend(accepted)
            raise RuntimeError('Time loop capacity retry limit')
        sync();elapsed=time.perf_counter()-start
        assert completed==steps
        self.completed_rows.extend(accepted);self.batch_history.append(dict(accepted_steps=completed,end_time_s=self.t,online_s=elapsed))
        self.statistics.update(time_batch_launches=self.batch_launches,time_loop_device=True,device_step_exceptions=self.exceptions,
            device_step_graph_rebuilds=self.graph_rebuilds)
        return accepted,elapsed
