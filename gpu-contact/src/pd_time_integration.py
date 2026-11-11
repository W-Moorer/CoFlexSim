"""Shared PD time operators. External FEM references remain native Abaqus.

HHT uses endpoint multiplier forces, not HHT-averaged multipliers:
 M a1 + (1+alpha)(fint1-fext1) - alpha(fint0-fext0) - G1.T lambda1 = 0.
The PD potential is divided by w=1+alpha, leaving its material majorizer
unchanged. Therefore physical lambda1 = w * lambda_QP.
"""
import numpy as np


class PDTimeIntegration:
    def __init__(self,policy):
        self.scheme=policy.get('time_integrator','backward_euler')
        if self.scheme not in ('backward_euler','hht'):raise ValueError('Unknown PD time integrator')
        self.alpha=float(policy.get('hht_alpha',-.05)) if self.scheme=='hht' else 0.
        if not -1/3<=self.alpha<=0:raise ValueError('HHT alpha must be between -1/3 and 0')
        self.beta=(1-self.alpha)**2/4;self.gamma=.5-self.alpha
        self.force_scale=1+self.alpha if self.scheme=='hht' else 1.
        self.mass_factor=1/(self.beta*self.force_scale) if self.scheme=='hht' else 1.
        self.acceleration=None;self.coordinate_acceleration=None
        self.previous_contact=None;self.previous_contact_moments=None

    def begin(self,sim,t):
        self.step_time=t
        h=sim.dt;self.old_velocity=sim.carrier_velocity.copy();self.old_qdot=sim.qdot.copy()
        force,loads=sim.forces(t);self.current_force=force.copy();self.current_loads=list(loads)
        self.old_carrier=sim.carrier.copy();self.old_x=sim.x.copy();self.old_q=sim.q.copy()
        if self.acceleration is None:
            # Explicitly matches the native IMPACT=NO reference's INITIAL=NO.
            self.acceleration=np.zeros_like(sim.carrier);self.coordinate_acceleration=np.zeros_like(sim.q)
        self.old_acceleration=self.acceleration.copy();self.old_coordinate_acceleration=self.coordinate_acceleration.copy()
        if self.scheme=='backward_euler':return sim.carrier+h*self.old_velocity,force,loads
        pred=sim.carrier+h*self.old_velocity+h*h*(.5-self.beta)*self.old_acceleration
        old_force,old_loads=sim.forces(sim.t)
        # Old material forces are spatial nodal vectors. Pull them back at the
        # CURRENT configuration through the same external virtual-work map.
        internal=sim.material_gradient_world(sim.q)
        self.old_nonconstraint=internal-old_force
        c=self.alpha/self.force_scale
        return pred,force+c*self.old_nonconstraint,[new-c*old for new,old in zip(loads,old_loads)]

    def finish(self,sim,pred):
        h=sim.dt;chi=sim.mapping(sim.q,carrier=True,jacobian=False)[0]
        if self.scheme=='backward_euler':
            sim.qdot=(sim.q-self.old_q)/h;sim.v=(sim.x-self.old_x)/h
            sim.carrier_velocity=(chi-self.old_carrier)/h
            self.acceleration=(sim.carrier_velocity-self.old_velocity)/h
        else:
            self.acceleration=(chi-pred)/(self.beta*h*h)
            sim.carrier_velocity=self.old_velocity+h*((1-self.gamma)*self.old_acceleration+self.gamma*self.acceleration)
            # Dynamic coordinate rates drive surface prediction; algebraic
            # residual rates are displacement differences, not extra inertia.
            qa=(sim.q-self.old_q-h*self.old_qdot-h*h*(.5-self.beta)*self.old_coordinate_acceleration)/(self.beta*h*h)
            sim.qdot=(sim.q-self.old_q)/h
            for i,b in enumerate(sim.bodies):
                lo=sim.q_offsets[i];idx=slice(lo,lo+b.rd)
                sim.qdot[idx]=self.old_qdot[idx]+h*((1-self.gamma)*self.old_coordinate_acceleration[idx]+self.gamma*qa[idx])
            self.coordinate_acceleration=qa
            residual_velocity=((sim.x-chi)-(self.old_x-self.old_carrier))/h
            sim.v=sim.carrier_velocity+residual_velocity
        sim.carrier=chi

    def guide_terms(self,spec,index,q,h):
        """Incremental potential, gradient and tangent of a grounded reference spring/dashpot."""
        k=float(spec.get('guide_stiffness',0.));c=float(spec.get('guide_damping',0.))
        if not (np.isfinite(k+c) and k>=0 and c>=0):raise ValueError('Invalid passive guide coefficients')
        if k==0 and c==0:return 0.,0.,0.
        old=self.old_q[index];shift=q-old
        target_rate=float(spec.get('guide_target_velocity',0.))
        extension=old-float(spec.get('guide_rest_coordinate',0.))-target_rate*self.step_time
        if self.scheme=='hht':
            rate=self.gamma/(self.beta*h)
            v=self.old_qdot[index];a=self.old_coordinate_acceleration[index]
            vbase=v+h*(1-self.gamma)*a-rate*(h*v+h*h*(.5-self.beta)*a)
            linear=k*extension+c*(vbase-target_rate)-self.alpha/self.force_scale*(k*(extension+h*target_rate)+c*(v-target_rate))
        else:rate=1/h;linear=k*extension-c*target_rate
        tangent=k+c*rate
        return shift*(linear+.5*tangent*shift),linear+tangent*shift,tangent

    def contact_impulse(self,nodal,moments,h):
        if self.previous_contact is None:self.previous_contact=np.zeros_like(nodal)
        if self.previous_contact_moments is None:self.previous_contact_moments=np.zeros_like(moments)
        weight=self.gamma if self.scheme=='hht' else 1.
        impulse=h*((1-weight)*self.previous_contact+weight*nodal)
        angular=h*((1-weight)*self.previous_contact_moments+weight*moments)
        self.previous_contact=nodal.copy();self.previous_contact_moments=np.asarray(moments).copy()
        return impulse,angular

    def record(self):
        return dict(scheme=self.scheme,alpha=self.alpha if self.scheme=='hht' else None,
            beta=self.beta if self.scheme=='hht' else None,gamma=self.gamma if self.scheme=='hht' else None,
            inertial_mass_factor=self.mass_factor,QP_multiplier_to_physical_force=self.force_scale,
            initial_acceleration='zero_explicitly_matching_native_IMPACT_NO',
            contact_force_history='endpoint_multiplier_not_HHT_weighted',
            contact_impulse='Newmark_gamma_quadrature' if self.scheme=='hht' else 'backward_Euler_endpoint',
            separate_impact_velocity_projection=False,
            residual_velocity='dynamic_Newmark_plus_algebraic_displacement_difference' if self.scheme=='hht' else 'displacement_difference')
