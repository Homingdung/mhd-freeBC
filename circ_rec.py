# 2D MHD free boundary problem
# use gmsh
# 2: interface
# 1: outer BC
from firedrake import *
from firedrake.pyplot import tricontourf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# solver parameter
lu = {
	 "mat_type": "aij",
	 "snes_type": "newtonls",
	 "ksp_type":"preonly",
	 "pc_type": "lu",
	 "pc_factor_mat_solver_type":"mumps"
}
sp = lu
# physical parameter
s = Constant(1)
nu = Constant(1)
eta = Constant(1)

# time parameter
t = Constant(0)
dt = Constant(0.1)
T = 10.0

def scross(x, y):
    return x[0]*y[1] - x[1]*y[0]


def vcross(x, y):
    return as_vector([x[1]*y, -x[0]*y])


def scurl(x):
    return x[1].dx(0) - x[0].dx(1)


def vcurl(x):
    return as_vector([x.dx(1), -x.dx(0)])


def acurl(x):
    return as_vector([
                     x[2].dx(1),
                     -x[2].dx(0),
                     x[1].dx(0) - x[0].dx(1)
                     ])

def plot_solution(u_h, time=None, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=[5, 4])
    if vmin is None or vmax is None:
        levels = None
    else:
        levels = np.linspace(vmin, vmax, 11)
    cs = tricontourf(u_h, axes=ax, levels=levels)
    ax.set_aspect('equal')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    if time is not None:
        ax.set_title(f'$t={time}$')
    cbar = fig.colorbar(cs)

#mesh = UnitDiskMesh(2)
mesh = Mesh("mesh/circle_in_rect.msh")
x, y = SpatialCoordinate(mesh)

DG0 = FunctionSpace(mesh, "DG", 0)
indicator_function = Function(DG0).interpolate(conditional((x -0.5)**2 + (y -0.5)**2 < 1/16, 1, 0))
mesh.mark_entities(indicator_function, 999)
mesh = RelabeledMesh(mesh, [indicator_function], [999])
mesh2 = Submesh(mesh, 2, 999)

# function space
Vg = VectorFunctionSpace(mesh, "CG", 2)
Vn = FunctionSpace(mesh, "CG", 1)
Vb = VectorFunctionSpace(mesh, "CG", 1)

#(P2, P1, P1)
Z = MixedFunctionSpace([Vg, Vn, Vb])
z = Function(Z)
z_prev = Function(Z)
(u, p, B) = split(z)
(ut, pt, Bt) = split(TestFunction(Z))
(up, pp, Bp) = split(z_prev)

#initial condition
A = 1.0        
sigma = 0.1   
x0, y0 = 0.5, 0.5
psi = A * exp(-((x-x0)**2 + (y-y0)**2)/sigma**2)
u_init = as_vector([psi.dx(1), -psi.dx(0)])
B0 = 1.0
B_init = as_vector([B0, 0.0])

z_prev.sub(0).interpolate(u_init)
z_prev.sub(2).interpolate(B_init)

z.assign(z_prev)

# domain as a function
V_coords = VectorFunctionSpace(mesh, 'CG', 1)
V_coords_mesh2 = VectorFunctionSpace(mesh2, 'CG', 1)
w_vel = Function(V_coords)
w_vel_mesh2 = Function(V_coords_mesh2)
 
# velocity of the boundary
#v = as_vector([x, y])*exp(-t)
v = z_prev.sub(0)
bc_vel = Function(V_coords).interpolate(v)
w_trial, w_test = TrialFunction(V_coords), TestFunction(V_coords)
bc_move_mesh = DirichletBC(V_coords, bc_vel, (2, )) # interface
BB = inner(grad(w_trial), grad(w_test))*dx - Constant(0)*w_test[0]*dx


mm_problem = LinearVariationalProblem(lhs(BB), rhs(BB), w_vel, bcs=bc_move_mesh)
mm_solver = LinearVariationalSolver(mm_problem)

dx = Measure("dx", domain=mesh)
F = (
    # equation for u
    + inner((u-up)/dt, ut) * dx
    + nu * inner(grad(u), grad(ut)) * dx
    + inner(dot(grad(u), (u - w_vel)), ut) * dx # w is the domain velocity
    - inner(u, ut * div(w_vel)) * dx
    - inner(p, div(ut)) * dx
    + s*inner(vcross(Bp, scurl(B)), ut) * dx # vcross(x, y) x is vector, y is scalar
    + 1/dt * inner(div(u), div(ut)) * dx
#- inner(f,  ut) * dx
    
    #equation for p
    - inner(div(u), pt) * dx
    # equation for B
    - s * inner((B-Bp)/dt, Bt) * dx
    - s * eta * inner(scurl(B), scurl(Bt)) * dx
    - s * eta * inner(div(B), div(Bt)) * dx
    + s * inner(scross(u, Bp), scurl(Bt)) * dx
    + s * inner(dot(grad(B), w_vel), Bt) * dx
    + s * inner(B, Bt * div(w_vel)) * dx
#    + s * inner(g, Bt) * dx
)

w_vel_pb = Function(Vg).interpolate(w_vel)
bcs = [
    DirichletBC(Z.sub(0), 0, (1, )), #outer boundary
    DirichletBC(Z.sub(0), w_vel_pb, (2, )), # interface
    DirichletBC(Z.sub(1), 0, (2, )), # interface
    DirichletBC(Z.sub(2), 0, (1, ))
]

pb = NonlinearVariationalProblem(F, z, bcs)
time_stepper = NonlinearVariationalSolver(pb, solver_parameters = sp)

pvd = VTKFile("output/mhd-free.pvd")
(u_, p_, B_) = z.subfunctions
u_.rename("Velocity")
p_.rename("Pressure")
B_.rename("MagneticField")

B_history = []
while (float(t) < float(T-dt)+1.0e-10):
    t.assign(t+dt)
    if mesh.comm.rank == 0:
        print(f"solving for t={float(t)}")
    mm_solver.solve()
    w_vel_mesh2.interpolate(w_vel)
    mesh2.coordinates.assign(mesh2.coordinates + dt * w_vel_mesh2) # solve for domain

    time_stepper.solve() # solve for PDE 
    pvd.write(*z.subfunctions, time = float(t))
    z_prev.assign(z)
    # update the interface velocity
    bc_vel = Function(V_coords).interpolate(z_prev.sub(0))
    print(norm(bc_vel, "L2"))
    #triplot(mesh2) # see whether the mesh is moving
    #plt.show()
    B_history.append(z.sub(2).copy(deepcopy=True))
     
import numpy as np
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(5,4))
ax.set_aspect('equal')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

cs = tricontourf(B_history[0], axes=ax)

def update(frame):
    ax.clear()
    ax.set_aspect('equal')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(f"t = {frame*float(dt):.2f}")
    return tricontourf(B_history[frame], axes=ax)

ani = FuncAnimation(fig, update, frames=len(B_history), interval=100)
ani.save("output/cir_in_rect.gif", writer="pillow", fps=10)
plt.show()
