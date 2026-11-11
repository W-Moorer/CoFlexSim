"""Fixed-capacity graph for current BVH, overlap, pressure and guard geometry.

Every replay recomputes current geometry. Capacities are storage bounds only;
overflows are rejected and rebuilt. No history-dependent geometric freezing.
The returned views are consumed before the next build on the same workspace.
"""
from pathlib import Path
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as csp

class PairGeometryGraph:
    def __init__(self,geometry,slave,master,capacity=8192,slots=128):
        self.geom=geometry;self.slave=slave;self.master=master;self.cap=capacity;self.slots=slots
        root=Path(__file__).parent
        self.module=cp.RawModule(code=(root/'geometry_graph.cu').read_text())
        self.k={n:self.module.get_function(n) for n in ('pack_pairs','dense_support','geometry_report')}
        module=cp.RawModule(code='#define DYNAMIC_NW\n'+(root/'geometry_assembly.cu').read_text())
        self.a={n:module.get_function(n) for n in ('support_sum','count_entries','assemble_rows')}
        self.allocate()

    @staticmethod
    def launch(kernel,n,args):
        if n:kernel(((n+127)//128,),(128,),args)

    def allocate(self):
        g=self.geom;cap=self.cap;self.fs=g.faces[self.slave];self.fm=g.faces[self.master];nf=len(self.fs)
        self.x=cp.zeros((g.size//3,3));self.S=cp.zeros((cap,3,3));self.M=cp.zeros_like(self.S)
        self.qlo=cp.empty((nf,3));self.qhi=cp.empty_like(self.qlo);self.out=cp.empty((nf,self.slots),dtype=cp.int32)
        self.qcount=cp.zeros(nf,dtype=cp.int32);self.poff=cp.zeros(nf+1,dtype=cp.int32);self.qerror=cp.zeros(1,dtype=cp.int32)
        self.owners=cp.zeros(cap,dtype=cp.int32);self.mids=cp.zeros_like(self.owners);self.nodes=cp.zeros((cap,6),dtype=cp.int32)
        self.area=cp.zeros((cap,2,3));self.weighted=cp.zeros_like(self.area);self.points=cp.empty((cap,2,3,3));self.info=cp.empty((cap,2,6))
        self.samples=cp.zeros((cap,2),dtype=cp.int32);self.vc=cp.zeros(cap,dtype=cp.int32);self.voff=cp.zeros(cap+1,dtype=cp.int32)
        self.vg=cp.empty((cap,12));self.vp=cp.empty((cap,12,3));self.partial=cp.empty((cap,6,18));self.vertex=cp.empty((cap,12,18))
        self.selected=cp.arange(cap,dtype=cp.int32);self.coverage=cp.zeros((2,nf));self.sumA=cp.zeros(g.maxnodes);self.sumG=cp.zeros(g.maxnodes)
        self.mask=cp.empty(g.maxnodes,dtype=cp.bool_);self.prefix=cp.empty(g.maxnodes,dtype=cp.int32);self.lookup=cp.empty_like(self.prefix)
        rows=g.maxnodes+12*cap;nnz=cap*270
        self.counts=cp.zeros(rows,dtype=cp.int32);self.ptr=cp.zeros(rows+1,dtype=cp.int32);self.cursor=cp.empty(rows,dtype=cp.int32)
        self.cols=cp.empty(nnz,dtype=cp.int32);self.vals=cp.empty(nnz);self.gaps=cp.empty(rows);self.keys=cp.empty(rows,dtype=cp.int64)
        self.aerror=cp.zeros(1,dtype=cp.int32);self.report=cp.empty(7,dtype=cp.int32);self.graphs={}

    def dispatch(self,derivatives):
        g=self.geom;t=g.trees[self.master];nf=len(self.fs);cap=self.cap;nw=self.prefix[-1:]
        sm=self.x[g.offsets[self.slave]:g.offsets[self.slave+1]];mm=self.x[g.offsets[self.master]:g.offsets[self.master+1]]
        self.launch(t.kernels['triangle_boxes'],len(t.faces),(mm,t.faces,t.tlo,t.thi,np.int32(len(t.faces))))
        self.launch(t.kernels['leaf_boxes'],len(t.leaves),(t.nodes,t.order,t.leaves,np.int32(len(t.leaves)),t.tlo,t.thi,t.lo,t.hi))
        for ids in t.levels:self.launch(t.kernels['inner_boxes'],len(ids),(t.nodes,ids,np.int32(len(ids)),t.lo,t.hi))
        self.out.fill(-1);self.qcount.fill(0);self.qerror.fill(0)
        self.launch(t.kernels['triangle_boxes'],nf,(sm,self.fs,self.qlo,self.qhi,np.int32(nf)))
        self.launch(t.kernels['query_boxes'],nf,(t.nodes,t.order,t.lo,t.hi,t.tlo,t.thi,self.qlo,self.qhi,np.int32(nf),np.float64(g.radius),np.int32(self.slots),self.out,self.qcount,self.qerror))
        cp.cumsum(self.qcount,dtype=cp.int32,out=self.poff[1:]);self.S.fill(0);self.M.fill(0)
        self.launch(self.k['pack_pairs'],nf,(self.out,self.qcount,self.poff,np.int32(nf),np.int32(self.slots),np.int32(cap),self.fs,self.fm,self.x,
            np.int32(g.offsets[self.slave]),np.int32(g.offsets[self.master]),self.owners,self.mids,self.nodes,self.S,self.M,self.qerror))
        self.samples.fill(0);self.vc.fill(0)
        g.kernel.values_kernel(((cap+63)//64,),(64,),(self.S,self.M,np.int32(cap),np.float64(g.radius),np.float64(g.band),np.int32(g.asymmetric),
            self.area,self.weighted,self.points,self.info,self.samples,self.vc,self.vg,self.vp))
        if derivatives:g.kernel.derivative_kernel((cap,),(32,),(self.S,self.M,self.selected,np.int32(cap),np.float64(g.radius),np.float64(g.band),np.int32(g.asymmetric),self.samples,self.partial,self.vertex))
        self.coverage.fill(0);self.sumA.fill(0);self.sumG.fill(0)
        self.launch(self.a['support_sum'],cap,(self.owners,self.nodes,self.samples,self.area,self.weighted,np.int32(cap),np.int32(nf),self.coverage,self.sumA,self.sumG))
        cp.greater(self.sumA,1e-24,out=self.mask);cp.cumsum(self.mask,dtype=cp.int32,out=self.prefix)
        paircode=self.slave*len(g.bodies)+self.master
        self.launch(self.k['dense_support'],g.maxnodes,(self.sumA,self.sumG,self.prefix,np.int32(g.maxnodes),self.lookup,self.gaps,self.keys,np.int64(paircode*g.maxnodes)))
        cp.cumsum(self.vc,dtype=cp.int32,out=self.voff[1:])
        if derivatives:
            self.counts.fill(0)
            self.launch(self.a['count_entries'],12*cap,(self.nodes,self.samples,self.lookup,np.int32(cap),nw,np.int32(12*cap),self.counts))
            cp.cumsum(self.counts,dtype=cp.int32,out=self.ptr[1:]);cp.copyto(self.cursor,self.ptr[:-1])
        self.aerror.fill(0)
        self.launch(self.a['assemble_rows'],cap,(self.owners,self.mids,self.nodes,self.samples,self.area,self.info,self.coverage,self.sumA,self.sumG,self.lookup,
            self.partial,self.vertex,self.vg,self.vp,self.vc,self.voff,self.x,np.int32(cap),np.int32(nf),nw,np.int32(g.maxfaces),np.int32(paircode),
            np.int64(g.guard_base),np.float64(g.guard_offset),np.float64(g.radius),self.cursor,self.ptr,self.cols,self.vals,self.gaps,self.keys,self.aerror,np.int32(derivatives)))
        self.k['geometry_report']((1,),(32,),(self.prefix,np.int32(g.maxnodes),self.voff,np.int32(cap),self.ptr,self.poff,np.int32(nf),self.qerror,self.aerror,self.samples,self.report,np.int32(derivatives)))

    def build(self,x,derivatives):
        cp.copyto(self.x,x)
        if derivatives not in self.graphs:
            self.dispatch(derivatives);cp.cuda.get_current_stream().synchronize()
            stream=cp.cuda.Stream(non_blocking=True)
            with stream:
                stream.begin_capture();self.dispatch(derivatives);self.graphs[derivatives]=stream.end_capture()
        self.graphs[derivatives].launch();pairs,nw,m,nnz,qerr,aerr,overlap=map(int,cp.asnumpy(self.report))
        if qerr in (1,3):
            if qerr==1:self.slots*=2
            if qerr==3:self.cap*=2
            if self.cap>131072 or self.slots>8192:raise MemoryError('GPU geometry graph capacity budget exceeded')
            self.allocate();return self.build(x,derivatives)
        if qerr:raise RuntimeError('GPU BVH graph stack overflow')
        if aerr:raise RuntimeError(f'GPU finite-overlap geometry failure {aerr}')
        G=csp.csr_matrix((self.vals[:nnz],self.cols[:nnz],self.ptr[:m+1]),shape=(m,self.geom.size)) if derivatives else None
        minimum=cp.min(self.gaps[nw:m])-self.geom.guard_offset if m>nw else cp.asarray(0.)
        return self.gaps[:m],G,self.keys[:m],minimum,nw,pairs,overlap
