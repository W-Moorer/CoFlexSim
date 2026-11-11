"""Exact, sealed host library on the declared contact surface; no online FE solve."""
import time
import json
from pathlib import Path
import numpy as np


class HostSurfaceLibrary:
    def __init__(self, n, tile_size, tiles, row_bound, offline_s):
        self.n = int(n)
        self.tile_size = int(tile_size)
        self.tiles = tiles
        self.row_bound = row_bound
        self.sealed = True
        self.statistics = dict(source='sealed_host_residual_tiles', sealed=True,
            full_surface_coverage=True, online_factor_solves=0,
            stored_bytes=sum(a.nbytes for a in tiles.values()) + row_bound.nbytes,
            offline_s=offline_s, block_requests=0, query_s=0.,
            maximum_absolute_entry=float(row_bound.max(initial=0.)))

    @classmethod
    def build(cls, source, tile_size=192, budget_bytes=2*2**30, batch=32, progress=None, file_path=None, source_identity=None, packed_memory=False):
        tick=time.perf_counter(); n=source.n; tiles={}; bound=np.zeros(n)
        required=sum(8*min(tile_size,n-i)*min(tile_size,n-j)
                     for j in range(0,n,tile_size) for i in range(0,j+1,tile_size))
        if required+bound.nbytes>budget_bytes:
            raise MemoryError(f'Exact host library needs {required+bound.nbytes} bytes')
        backing=None;offset=0;completed=0;prior_s=0.;state=None
        if packed_memory:
            if file_path is not None:raise ValueError('Choose one packed backing: memory or file')
            backing=np.empty(required//8,dtype=np.float64)
        if file_path is not None:
            path=Path(file_path).resolve()
            allowed=(Path(__file__).resolve().parents[1]/'data').resolve()
            if not path.is_relative_to(allowed):raise ValueError('Offline libraries must remain in this checkout data directory on F:')
            path.parent.mkdir(parents=True,exist_ok=True)
            catalog=path.with_suffix('.json')
            if path.exists():
                if not catalog.exists():raise ValueError('Existing library lacks a recoverable catalog')
                state=json.loads(catalog.read_text())
                if any(state.get(k)!=v for k,v in dict(n=n,tile_size=tile_size,file_bytes=required,source_identity=source_identity).items()):
                    raise ValueError('Offline library model identity or layout differs; do not overwrite')
                if path.stat().st_size!=required:raise ValueError('Offline library file size differs')
                if state['status']=='sealed':return cls.open(path,source_identity=source_identity)
                completed=int(state['completed_columns']);prior_s=float(state.get('offline_s',0.))
            else:
                state=dict(schema=1,n=n,tile_size=tile_size,file_bytes=required,source_identity=source_identity,
                    status='building',completed_columns=0,offline_s=0.)
            backing=np.memmap(path,mode='r+' if path.exists() else 'w+',dtype=np.float64,shape=(required//8,))
            cls._write_json(catalog,state)
        for j in range(0,n,tile_size):
            width=min(tile_size,n-j)
            for i in range(0,j+1,tile_size):
                shape=(min(tile_size,n-i),width);size=shape[0]*shape[1]
                if backing is None:tile=np.empty(shape)
                else:
                    tile=backing[offset:offset+size].reshape(shape);offset+=size
                tiles[i//tile_size,j//tile_size]=tile
            for k in range(j if j>=completed else j+width,j+width,batch):
                ids=np.arange(k,min(k+batch,j+width)); raw=source.columns(ids)
                assert raw.shape==(n,len(ids)) and np.isfinite(raw).all()
                for i in range(0,j+1,tile_size):
                    a=raw[i:min(i+tile_size,n)]
                    tiles[i//tile_size,j//tile_size][:,k-j:k-j+len(ids)]=a
            for i in range(0,j+1,tile_size):
                a=tiles[i//tile_size,j//tile_size]
                if i==j and j>=completed:
                    asym=np.max(np.abs(a-a.T)); scale=np.max(np.abs(a))
                    if asym>1e-8*max(scale,1e-30):raise ValueError('Nonreciprocal residual source')
                    a[:]=(a+a.T)*.5
                bound[i:i+a.shape[0]]=np.maximum(bound[i:i+a.shape[0]],np.max(np.abs(a),axis=1))
                bound[j:j+a.shape[1]]=np.maximum(bound[j:j+a.shape[1]],np.max(np.abs(a),axis=0))
                a.flags.writeable=False
            if file_path is not None and j>=completed:
                backing.flush()
                state.update(completed_columns=min(j+width,n),offline_s=prior_s+time.perf_counter()-tick)
                cls._write_json(catalog,state)
            if progress:progress(min(j+width,n),n)
        if file_path is not None:backing.flush()
        result=cls(n,tile_size,tiles,bound,prior_s+time.perf_counter()-tick)
        result.backing=backing
        if file_path is not None:
            backing.flags.writeable=False
            result.statistics.update(source='sealed_F_disk_residual_tiles',file_path=str(path),file_bytes=required,
                host_residency='OS-managed file pages; no whole-library GPU upload')
            result.catalog_state=state
            result.save_catalog()
        elif packed_memory:
            backing.flags.writeable=False
            result.statistics.update(source='sealed_RAM_residual_tiles',
                host_residency='Packed host RAM; no disk bank and no online FE solve')
        return result

    @staticmethod
    def _write_json(path,value):
        temporary=path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(json.dumps(value,indent=2),encoding='utf-8')
        temporary.replace(path)

    def save_catalog(self):
        if getattr(self,'backing',None) is None or 'file_path' not in self.statistics:return
        path=Path(self.statistics['file_path']);bounds=path.with_suffix('.npz')
        values=dict(row_bound=self.row_bound)
        if hasattr(self,'far_column_bound'):values['far_column_bound']=self.far_column_bound
        temporary=bounds.with_suffix('.tmp')
        with temporary.open('wb') as stream:np.savez(stream,**values)
        temporary.replace(bounds)
        self.catalog_state.update(status='sealed',completed_columns=self.n,offline_s=self.statistics['offline_s'],
            bounds_file=bounds.name,statistics=self.statistics)
        self._write_json(path.with_suffix('.json'),self.catalog_state)

    @classmethod
    def open(cls,file_path,source_identity=None):
        path=Path(file_path).resolve();state=json.loads(path.with_suffix('.json').read_text())
        if state['status']!='sealed':raise ValueError('Incomplete library cannot be used online')
        if state.get('source_identity')!=source_identity:raise ValueError('Offline model identity differs')
        if path.stat().st_size!=state['file_bytes']:raise ValueError('Offline library length differs')
        n,t=state['n'],state['tile_size'];backing=np.memmap(path,mode='r',dtype=np.float64,shape=(state['file_bytes']//8,))
        tiles={};offset=0
        for j in range(0,n,t):
            for i in range(0,j+1,t):
                shape=(min(t,n-i),min(t,n-j));size=shape[0]*shape[1]
                tiles[i//t,j//t]=backing[offset:offset+size].reshape(shape);offset+=size
        with np.load(path.with_suffix('.npz')) as bounds:
            result=cls(n,t,tiles,bounds['row_bound'],state['offline_s'])
            if 'far_column_bound' in bounds:result.far_column_bound=bounds['far_column_bound']
        result.backing=backing;result.catalog_state=state
        result.statistics.update(state['statistics'],loaded_from_existing_file=True,query_s=0.,block_requests=0)
        return result

    def block(self, rows, cols):
        tick=time.perf_counter(); rows=np.asarray(rows,dtype=np.int64).ravel(); cols=np.asarray(cols,dtype=np.int64).ravel()
        if np.any(rows<0) or np.any(rows>=self.n) or np.any(cols<0) or np.any(cols>=self.n):
            raise IndexError('Request outside designated surface library')
        out=np.empty((len(rows),len(cols))); t=self.tile_size
        for a in np.unique(rows//t):
            ri=np.flatnonzero(rows//t==a); r=rows[ri]%t
            for b in np.unique(cols//t):
                ci=np.flatnonzero(cols//t==b); c=cols[ci]%t
                tile=self.tiles[min(a,b),max(a,b)]
                out[np.ix_(ri,ci)]=(tile if a<=b else tile.T)[np.ix_(r,c)]
        self.statistics['block_requests']+=1;self.statistics['query_s']+=time.perf_counter()-tick
        return out

    def columns(self, ids):return self.block(np.arange(self.n),ids)

    def bounds_outside(self, resident, domain=None):
        """For each resident load column, max |R_ij| over nonresident rows."""
        tick=time.perf_counter();resident=np.asarray(resident,dtype=int)
        inside=np.zeros(self.n,dtype=bool);inside[resident]=True
        outside=~inside
        if domain is not None:
            allowed=np.zeros(self.n,dtype=bool);allowed[domain]=True;outside&=allowed
        t=self.tile_size;ii={};oo={};out=np.zeros(self.n)
        for start in range(0,self.n,t):
            ii[start//t]=np.flatnonzero(inside[start:start+t])
            oo[start//t]=np.flatnonzero(outside[start:start+t])
        for (a,b),tile in self.tiles.items():
            if len(oo[a]) and len(ii[b]):
                ids=b*t+ii[b]
                out[ids]=np.maximum(out[ids],np.max(np.abs(tile[np.ix_(oo[a],ii[b])]),axis=0))
            if a!=b and len(oo[b]) and len(ii[a]):
                ids=a*t+ii[a]
                out[ids]=np.maximum(out[ids],np.max(np.abs(tile[np.ix_(ii[a],oo[b])]),axis=1))
        self.statistics['bound_query_s']=self.statistics.get('bound_query_s',0.)+time.perf_counter()-tick
        return out

    def prepare_far_bounds(self, faces, domain=None):
        """Offline bound excluding each load node's guaranteed resident one-ring."""
        import scipy.sparse as sp
        tick=time.perf_counter();faces=np.asarray(faces,dtype=int)
        if self.n%3:raise ValueError('Geometry certificate requires 3 DOFs per node')
        edges=np.array([(int(i),int(j)) for f in faces for i in f for j in f],dtype=int)
        near=sp.coo_matrix((np.ones(len(edges),dtype=bool),(edges[:,0],edges[:,1])),shape=(self.n//3,self.n//3)).tocsr()
        allowed=np.ones(self.n,dtype=bool) if domain is None else np.isin(np.arange(self.n),domain)
        bound=np.zeros(self.n);t=self.tile_size
        for (a,b),tile in self.tiles.items():
            rows=np.arange(a*t,a*t+tile.shape[0]);cols=np.arange(b*t,b*t+tile.shape[1])
            mask=near[rows//3][:,cols//3].toarray()
            values=np.abs(tile).copy();values[mask]=0.
            values[~allowed[rows],:]=0.;values[:,~allowed[cols]]=0.
            bound[cols]=np.maximum(bound[cols],np.max(values,axis=0))
            if a!=b:bound[rows]=np.maximum(bound[rows],np.max(values,axis=1))
        # Round the arithmetic inequality outward, rather than fit a trajectory.
        bound=np.nextafter(bound,np.inf)
        self.statistics['geometry_bound_offline_s']=time.perf_counter()-tick
        if domain is None:
            self.far_column_bound=bound
            self.save_catalog()
        return bound

    def apply(self, force, rows=None):
        f=np.asarray(force); ids=np.flatnonzero(f)
        rows=np.arange(self.n) if rows is None else np.asarray(rows)
        out=np.zeros(len(rows))
        for k in range(0,len(ids),32):
            cols=ids[k:k+32];out+=self.block(rows,cols)@f[cols]
        return out
