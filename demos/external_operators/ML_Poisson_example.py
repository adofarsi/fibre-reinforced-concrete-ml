import torch
import torch.nn as nn

from firedrake import *

import matplotlib.pyplot as plt

class EncoderDecoder(nn.Module):
    def __init__(self, n):
        super(EncoderDecoder, self).__init__()

        # Define your model architecture
        m = int(n/10)
        self.encoder = nn.Linear(n + 1, m)
        self.hidden = nn.Linear(m, m)
        self.decoder = nn.Linear(m, n)

    def forward(self, x):
        x = torch.relu(self.encoder(x))
        x = torch.relu(self.hidden(x))
        return self.decoder(x)

N = 50
mesh = UnitSquareMesh(N, N)
V = FunctionSpace(mesh, "CG", 2)

u = TrialFunction(V)
v = TestFunction(V)
c = Constant(2.)
k = Function(V).assign(1.)

# Define the operator representing the machine learning model
model = EncoderDecoder(V.dim())
neural_operator = neuralnet(model, function_space=V)
f = neural_operator(c, k)

# Define the bilinear and linear forms
a = inner(k * grad(u), grad(v)) * dx
L = f*v*dx

# Set boundary conditions (homogeneous Dirichlet)
bcs = [DirichletBC(V, Constant(0.), "on_boundary")]

# Solve the Poisson problem:
#  - ∇.(k ∇u) = f  in Ω
#           u = 0  on ∂Ω
# where f is modelled by a neural network: (c, k) -> f(c, k)
uu = Function(V)
solve(a == L, uu, bcs=bcs, solver_parameters={"ksp_type": "preonly",
                                              "pc_type": "lu"})

# Plot solution
fig = tripcolor(uu, cmap="coolwarm")
plt.title("Solution")
plt.colorbar(fig)

# Plot ML model output
fig = tripcolor(assemble(f), cmap="coolwarm")
plt.title("Source f(c, k) (ML model)")
plt.colorbar(fig)

plt.show()