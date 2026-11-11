"""Finite-overlap, P1 pressure / P1 displacement surface contact.

Common-refinement polygons are clipped in each secondary triangle's plane.
Three-point subtriangle integration exactly integrates the frozen P1 products.
Constraints are normalized weak gaps; multipliers divided by support area are
pressure coefficients, NOT point forces at secondary nodes. This implements a
standard mortar-type discretization, not a novel theory or Abaqus replica.
Geometry is rebuilt during SQP; its tangent freezes projection and segmentation.
No facet extension, friction, edge-edge constraints, or CCD guarantee is added.
The optional overlap-biorthogonal dual basis is experimental: on curved/open
patches it can extrapolate a negative weak gap while all sampled gaps are open.
Such detected cases are rejected, never accepted as evidence of true contact.
"""
import numpy as np
import scipy.sparse as sp
from rtree import index


def barycentric_2d(points, triangle):
    ab=(np.asarray(points)-triangle[0])@np.linalg.inv((triangle[1:]-triangle[0]).T).T
    return np.column_stack((1.-ab.sum(axis=1),ab))


def clip_halfspace(polygon, coefficients, tolerance=2e-13):
    """Clip a convex polygon by a*x+b*y+c >= 0 in local dimensionless units."""
    if len(polygon)<3:return np.empty((0,2))
    out=[];previous=polygon[-1];fp=previous@coefficients[:2]+coefficients[2]+tolerance
    for current in polygon:
        fc=current@coefficients[:2]+coefficients[2]+tolerance
        if (fp>=0)!=(fc>=0):
            out.append(previous+(current-previous)*(fp/(fp-fc)))
        if fc>=0:out.append(current)
        previous=current;fp=fc
    if len(out)<3:return np.empty((0,2))
    return np.asarray(out)


def overlap_quadrature(secondary,main,radius,sampling='quadrature'):
    edge=secondary[1]-secondary[0];length=np.linalg.norm(edge)
    normal=np.cross(edge,secondary[2]-secondary[0]);area2=np.linalg.norm(normal)
    if length<=0 or area2<=1e-24:return None
    outward=normal/area2;normal=-outward
    em=np.cross(main[1]-main[0],main[2]-main[0]);am=np.linalg.norm(em)
    if am<=1e-24 or (em/am)@outward>-.25:return None
    e1=edge/length;e2=np.cross(outward,e1);frame=np.column_stack((e1,e2))/length
    S=(secondary-secondary[0])@frame;M=(main-secondary[0])@frame
    if abs(np.linalg.det((M[1:]-M[0]).T))<1e-12:return None
    # Affine barycentric functions evaluated at (0,0), (1,0), (0,1).
    probes=np.array([[0.,0.],[1.,0.],[0.,1.]])
    bm=barycentric_2d(probes,M)
    affine=np.vstack((bm[1]-bm[0],bm[2]-bm[0],bm[0]))
    polygon=S.copy()
    for j in range(3):polygon=clip_halfspace(polygon,affine[:,j])
    if len(polygon)<3:return None
    # Signed ray gap from the main surface towards the secondary surface.
    probe_x=secondary[0]+probes[:,0,None]*length*e1+probes[:,1,None]*length*e2
    probe_gap=np.einsum('ij,j->i',probe_x-bm@main,normal)/length
    gc=np.array([probe_gap[1]-probe_gap[0],probe_gap[2]-probe_gap[0],probe_gap[0]])
    for sign in [-1.,1.]:
        c=sign*gc;c[2]+=radius/length;polygon=clip_halfspace(polygon,c)
    if len(polygon)<3:return None
    if sampling=='vertices':
        # Each frozen triangle-pair gap is affine over this convex polygon.
        # Its minimum is attained at a vertex. Area lumps are bookkeeping,
        # never a claimed distributed contact pressure reconstruction.
        unique=[]
        for p in polygon:
            if not any(np.linalg.norm(p-q)<1e-12 for q in unique):unique.append(p)
        polygon=np.asarray(unique)
        if len(polygon)<3:return None
        weights=np.zeros(len(polygon))
        for i in range(1,len(polygon)-1):
            ids=[0,i,i+1];tri=polygon[ids]
            a=.5*abs(np.linalg.det((tri[1:]-tri[0]).T))*length**2
            weights[ids]+=a/3
        if weights.sum()<area2*1e-14:return None
        Ns=barycentric_2d(polygon,S);Nm=barycentric_2d(polygon,M)
        return Ns,Nm,weights,(Ns@secondary-Nm@main)@normal,normal,area2/2
    if sampling!='quadrature':raise ValueError(sampling)
    rule=np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
    points=[];weights=[]
    for i in range(1,len(polygon)-1):
        tri=polygon[[0,i,i+1]];a=.5*abs(np.linalg.det((tri[1:]-tri[0]).T))*length**2
        if a>area2*1e-14:points.extend(rule@tri);weights.extend([a/3]*3)
    if not points:return None
    points=np.asarray(points);weights=np.asarray(weights)
    Ns=barycentric_2d(points,S);Nm=barycentric_2d(points,M)
    if min(Ns.min(),Nm.min())<-1e-10:raise RuntimeError('Mortar point escaped finite triangle')
    gaps=((Ns@secondary-Nm@main)@normal)
    return Ns,Nm,weights,gaps,normal,area2/2


class MortarContact:
    def __init__(self,bodies,offsets,pairs,search_radius,pressure_basis='primal'):
        self.bodies=bodies;self.offsets=offsets;self.pairs=pairs;self.radius=search_radius
        if pressure_basis not in ['primal','dual']:raise ValueError(pressure_basis)
        self.pressure_basis=pressure_basis
        self.statistics={'calls':0,'candidate_triangle_pairs':0,'overlap_segments':0,'quadrature_points':0,'maximum_coverage_fraction':0.,'pressure_basis':pressure_basis,'maximum_dual_mass_condition':0.}

    def build(self,x,J):
        self.statistics['calls']+=1;assembled={}
        for slave,master in self.pairs:
            fs=self.bodies[slave].spec['contact_faces'];fm=self.bodies[master].spec['contact_faces']
            if not len(fs) or not len(fm):continue
            sm=x[self.offsets[slave]:self.offsets[slave+1]];mm=x[self.offsets[master]:self.offsets[master+1]]
            triangles=mm[fm];lower=triangles.min(axis=1);upper=triangles.max(axis=1)
            properties=index.Property();properties.dimension=3
            tree=index.Index(((i,tuple(np.r_[lo,hi]),None) for i,(lo,hi) in enumerate(zip(lower,upper))),properties=properties)
            for face in fs:
                secondary=sm[face];box=np.r_[secondary.min(axis=0)-self.radius,secondary.max(axis=0)+self.radius]
                candidates=sorted(tree.intersection(tuple(box)));coverage=0.;area=0.;segments=[]
                self.statistics['candidate_triangle_pairs']+=len(candidates)
                for m in candidates:
                    rule=overlap_quadrature(secondary,triangles[m],self.radius)
                    if rule is None:continue
                    Ns,Nm,w,g,n,area=rule;coverage+=w.sum()
                    self.statistics['overlap_segments']+=1;self.statistics['quadrature_points']+=len(w)
                    segments.append((m,rule))
                transform=np.eye(3)
                if segments and self.pressure_basis=='dual':
                    mass=sum(Ns.T@(w[:,None]*Ns) for _,(Ns,Nm,w,g,n,area) in segments)
                    lumped=mass.sum(axis=1);condition=np.linalg.cond(mass)
                    self.statistics['maximum_dual_mass_condition']=max(self.statistics['maximum_dual_mass_condition'],float(condition))
                    if condition>1e12:raise RuntimeError('Partial overlap dual mass is ill-conditioned; boundary aggregation required')
                    # Define duality on the ACTUAL finite overlap, not the whole
                    # secondary triangle: integral Psi_i N_j = a_i delta_ij.
                    transform=np.linalg.solve(mass,np.diag(lumped))
                for m,(Ns,Nm,w,g,n,area) in segments:
                    Psi=Ns@transform
                    a=Psi.T@w;gg=Psi.T@(w*g);D=Psi.T@(w[:,None]*Ns);M=Psi.T@(w[:,None]*Nm)
                    point=Psi.T@(w[:,None]*(Nm@triangles[m]))
                    for k,node in enumerate(face):
                        key=(slave,master,int(node));r=assembled.setdefault(key,{'area':0.,'gap':0.,'entries':{},'point':np.zeros(3),'normal':np.zeros(3),'faces':{},'min_sample_gap':np.inf,'max_sample_gap':-np.inf})
                        r['area']+=a[k];r['gap']+=gg[k];r['point']+=point[k];r['normal']+=a[k]*n
                        r['faces'][m]=r['faces'].get(m,0.)+a[k]
                        r['min_sample_gap']=min(r['min_sample_gap'],float(g.min()));r['max_sample_gap']=max(r['max_sample_gap'],float(g.max()))
                        for node2,value in zip(np.r_[face+self.offsets[slave],fm[m]+self.offsets[master]],np.r_[D[k],-M[k]]):
                            node2=int(node2);r['entries'][node2]=r['entries'].get(node2,np.zeros(3))+value*n
                if area:
                    fraction=coverage/area;self.statistics['maximum_coverage_fraction']=max(self.statistics['maximum_coverage_fraction'],float(fraction))
                    if fraction>1.+1e-7:raise RuntimeError('Multiple main sheets overlap a secondary triangle; ray visibility partition required')
            tree.close()
        rows=[];cols=[];values=[];gaps=[];records=[]
        for key,r in sorted(assembled.items()):
            if r['area']<=1e-24:continue
            if self.pressure_basis=='dual' and r['gap']/r['area']<-1e-10 and r['min_sample_gap']>1e-10:
                raise RuntimeError('Dual gap extrapolation on an entirely open sampled patch; curved-contact validity gate failed')
            row=len(gaps);ids=np.array(sorted(r['entries']),int);vec=np.array([r['entries'][i] for i in ids])/r['area']
            rows.extend([row]*3*len(ids));cols.extend((3*ids[:,None]+np.arange(3)).ravel());values.extend(vec.ravel())
            gaps.append(r['gap']/r['area']);records.append({'key':key,'ids':np.r_[key[2]+self.offsets[key[0]],ids],
                'normal':r['normal']/r['area'],'master_face':max(r['faces'],key=r['faces'].get),'point':r['point']/r['area'],
                'support_area':float(r['area']),'min_sample_gap':r['min_sample_gap'],'max_sample_gap':r['max_sample_gap']})
        Gx=sp.csr_matrix((values,(rows,cols)),shape=(len(gaps),x.size))
        return np.asarray(gaps),Gx,(Gx@J).tocsr(),records
