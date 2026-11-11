"""Parallel-kernel variant of v4; shared equations and acceptance are unchanged."""
import os
from device_loop_simulator import DeviceLoopSurfaceCMS
from parallel_modal import ParallelModalKernels
from aliased_working import AliasedWorkingSet
from device_cached_normal import DeviceCachedNormal
from fused_region_contact import FusedRegionContact

class ParallelLoopSurfaceCMS(DeviceLoopSurfaceCMS):
    def rebuild(self,refresh=True):
        if os.environ.get('CONTACT_DEVICE_CACHE','1')=='1' and not isinstance(self.geometry,DeviceCachedNormal):
            self.geometry=DeviceCachedNormal(self.bodies,self.scene)
        if os.environ.get('CONTACT_PARALLEL_MODAL','1')=='1' and not isinstance(self.modal_kernels,ParallelModalKernels):
            self.modal_kernels=ParallelModalKernels()
        if os.environ.get('CONTACT_ALIAS_MAPS','1')=='1' and not isinstance(self.graph_working,AliasedWorkingSet):
            self.graph_working=AliasedWorkingSet(self)
        if os.environ.get('CONTACT_FUSE_PREPARE','1')=='1' and not isinstance(self.resident,FusedRegionContact):
            self.resident=FusedRegionContact(self)
        return super().rebuild(refresh)
