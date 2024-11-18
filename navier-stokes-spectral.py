import numpy as np
import matplotlib.pyplot as plt
from adios2 import Stream, Adios
#from mpi4py import MPI

"""
Create Your Own Navier-Stokes Spectral Method Simulation (With Python)
Philip Mocz (2023), @PMocz

Simulate the Navier-Stokes equations (incompressible viscous fluid) 
with a Spectral method

v_t + (v.nabla) v = nu * nabla^2 v + nabla P
div(v) = 0

"""


def poisson_solve( rho, kSq_inv ):
	""" solve the Poisson equation, given source field rho """
	V_hat = -(np.fft.fftn( rho )) * kSq_inv
	V = np.real(np.fft.ifftn(V_hat))
	return V


def diffusion_solve( v, dt, nu, kSq ):
	""" solve the diffusion equation over a timestep dt, given viscosity nu """
	v_hat = (np.fft.fftn( v )) / (1.0+dt*nu*kSq)
	v = np.real(np.fft.ifftn(v_hat))
	return v


def grad(v, kx, ky):
	""" return gradient of v """
	v_hat = np.fft.fftn(v)
	dvx = np.real(np.fft.ifftn( 1j*kx * v_hat))
	dvy = np.real(np.fft.ifftn( 1j*ky * v_hat))
	return dvx, dvy


def div(vx, vy, kx, ky):
	""" return divergence of (vx,vy) """
	dvx_x = np.real(np.fft.ifftn( 1j*kx * np.fft.fftn(vx)))
	dvy_y = np.real(np.fft.ifftn( 1j*ky * np.fft.fftn(vy)))
	return dvx_x + dvy_y


def curl(vx, vy, kx, ky):
	""" return curl of (vx,vy) """
	dvx_y = np.real(np.fft.ifftn( 1j*ky * np.fft.fftn(vx)))
	dvy_x = np.real(np.fft.ifftn( 1j*kx * np.fft.fftn(vy)))
	return dvy_x - dvx_y


def apply_dealias(f, dealias):
	""" apply 2/3 rule dealias to field f """
	f_hat = dealias * np.fft.fftn(f)
	return np.real(np.fft.ifftn( f_hat ))


def main():
	""" Navier-Stokes Simulation """
	#comm = MPI.COMM_WORLD

	# Simulation parameters
	N         = 400     # Spatial resolution                   done
	t         = 0       # current time of the simulation       done
	tEnd      = 1       # time at which simulation ends        done
	dt        = 0.001   # timestep                             done
	tOut      = 0.01    # draw frequency                       done
	nu        = 0.001   # viscosity                            done
	plotRealTime = True # switch on for plotting as the simulation goes along
	
	# Domain [0,1] x [0,1]
	L = 1    # done 
	xlin = np.linspace(0,L, num=N+1)  # Note: x=0 & x=1 are the same point! done
	xlin = xlin[0:N]                  # chop off periodic point             done
	xx, yy = np.meshgrid(xlin, xlin)
	
	# Intial Condition (vortex)
	vx = -np.sin(2*np.pi*yy)       
	vy =  np.sin(2*np.pi*xx*2) 
	
	# Fourier Space Variables
	klin = 2.0 * np.pi / L * np.arange(-N/2, N/2) # done
	kmax = np.max(klin) # done
	kx, ky = np.meshgrid(klin, klin) # done
	kx = np.fft.ifftshift(kx) # done
	ky = np.fft.ifftshift(ky) # done
	kSq = kx**2 + ky**2       # done
	kSq_inv = 1.0 / kSq       # done
	kSq_inv[kSq==0] = 1       # done
	
	# dealias with the 2/3 rule
	dealias = (np.abs(kx) < (2./3.)*kmax) & (np.abs(ky) < (2./3.)*kmax) # done
	
	# number of timesteps
	Nt = int(np.ceil(tEnd/dt))  # done


	
	# prep figure
	fig = plt.figure(figsize=(4,4), dpi=80)
	outputCount = 1
	

	
	
	# adios = Adios("adios2.xml", mpi.comm_app)
	# io = adios.declare_io("uncompressed error")
	# fout = Stream(io, "Navier-stokes.bp", "w")
	#Main Loop
	# you can change the name of this to be whatever you want it to be
	with Stream ("Navier-stokes.bp", "w") as s:
		for _ in s.steps(Nt):    #i in range(Nt):

			# Advection: rhs = -(v.grad)v
			dvx_x, dvx_y = grad(vx, kx, ky) # done 
			dvy_x, dvy_y = grad(vy, kx, ky) # done
			
			rhs_x = -(vx * dvx_x + vy * dvx_y) # done
			rhs_y = -(vx * dvy_x + vy * dvy_y) # done
			
			rhs_x = apply_dealias(rhs_x, dealias) # done
			rhs_y = apply_dealias(rhs_y, dealias) # done

			vx += dt * rhs_x
			vy += dt * rhs_y
		
			# Poisson solve for pressure
			div_rhs = div(rhs_x, rhs_y, kx, ky) # done
			P = poisson_solve( div_rhs, kSq_inv ) # done 
			dPx, dPy = grad(P, kx, ky) # done
		
			# Correction (to eliminate divergence component of velocity)
			vx += - dt * dPx
			vy += - dt * dPy
		
			# Diffusion solve (implicit)
			vx = diffusion_solve( vx, dt, nu, kSq ) # done
			vy = diffusion_solve( vy, dt, nu, kSq ) # done
		
			# vorticity (for plotting) 
			wz = curl(vx, vy, kx, ky) # done
		
			# update time
			# print(type(wz))
		
			t += dt
			
			
			
			wz = curl(vx, vy, kx, ky)
			

			plotThisTurn = False


			# this can be deleled or changed 
			# this is here only becuase I write out a lot of data and I do not want write out so much on my computer
			if t >= 1:
				# making it contiguous array (C-order) in memory for wrghting it out in memory 
				wz_contiguous = np.ascontiguousarray(wz)
				dPx_contiguous = np.ascontiguousarray(dPx)
				dPy_contiguous = np.ascontiguousarray(dPy)
				div_rhs_contiguous = np.ascontiguousarray(div_rhs)
				P_contiguous = np.ascontiguousarray(P)
				vx_contiguous = np.ascontiguousarray(vx)
				vy_contiguous = np.ascontiguousarray(vy)	
				rhs_x_contiguous = np.ascontiguousarray(rhs_x)
				rhs_y_contiguous = np.ascontiguousarray(rhs_y)
				dvx_x_contiguous = np.ascontiguousarray(dvx_x)
				dvy_x_contiguous = np.ascontiguousarray(dvy_x)
				dealias = np.array(dealias, dtype=np.float64)
				dealias__contiguous  = np.ascontiguousarray(dealias)

				"""write out adios vars here"""
				s.write("curl", wz_contiguous, [400, 400], (0, 0), [400, 400])	
				s.write("gradient of V dPx", dPx_contiguous, [400,400] , (0,0), [400,400] )
				s.write("gradient of V dPy", dPy_contiguous, [400,400] , (0,0), [400,400] )
				s.write("divergence", div_rhs_contiguous, [400,400] , (0,0), [400,400] )
				s.write("pressure", P_contiguous, [400,400] , (0,0), [400,400] )
				s.write("Velocity X", vx_contiguous, [400,400], (0,0), [400,400])
				s.write("Velocity Y", vy_contiguous, [400,400], (0,0), [400,400])
				s.write("RHS X", rhs_x_contiguous, [400,400], (0,0), [400,400])
				s.write("RHS Y", rhs_y_contiguous, [400,400], (0,0), [400,400])
				s.write("second derivative Velocity X", dvx_x_contiguous, [400,400], (0,0), [400,400])
				s.write("second derivative Velocity Y", dvy_x_contiguous, [400,400], (0,0), [400,400])
				s.write("delta_T", dt)
				s.write("Spatial resolution",N)
				s.write("time",t)
				s.write("tEnd",tEnd)
				s.write("draw frequency", dt)
				s.write("viscosity",nu)
				s.write("NT",Nt)
				s.write("xlength",xlin)
				s.write("xx",xx)
				s.write("yy",yy)
				s.write("L",L)
				s.write("klin", klin, [len(klin)], [0], [len(klin)])
				s.write("kmax",kmax)
				s.write("kx",kx, [400,400], (0,0), [400,400])
				s.write("ky",ky, [400,400], (0,0), [400,400])
				s.write("kSq_inv",kSq_inv, [400,400], (0,0), [400,400])
				s.write("dealias", dealias__contiguous, [400,400], (0,0), [400,400])



			print(f"The time: {t}")
			
			
			
			# plot in real time
			
			if t + dt > outputCount*tOut:
				plotThisTurn = True
			if (plotRealTime and plotThisTurn) or (s == Nt-1):
			
				plt.cla()
				plt.imshow(wz, cmap = 'RdBu')
				plt.clim(-20,20)
				ax = plt.gca()
				ax.invert_yaxis()
				ax.get_xaxis().set_visible(False)
				ax.get_yaxis().set_visible(False)	
				ax.set_aspect('equal')	
				plt.pause(0.001)
				outputCount += 1
			
			
	# Save figure
	print("Just wrote it to this Navier-stokes.bp")
	
	plt.savefig('navier-stokes-spectral.png',dpi=240)
	plt.show()

	#MPI.Finalize()
	return 0
	


if __name__== "__main__":
  main()