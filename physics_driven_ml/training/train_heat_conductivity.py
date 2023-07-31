import os
import argparse
import functools

import torch
import torch.optim as optim
import torch.autograd as torch_ad

from tqdm.auto import tqdm, trange

from torch.utils.data import DataLoader

from firedrake import *
from firedrake_adjoint import *
from firedrake.ml.pytorch import torch_operator

from physics_driven_ml.dataset_processing import PDEDataset, BatchedElement
from physics_driven_ml.models import EncoderDecoder, CNN, ResNet
from physics_driven_ml.utils import ModelConfig, get_logger
from physics_driven_ml.evaluation import evaluate
import matplotlib.pyplot as plt

import numpy as np


def train(model, config: ModelConfig,
          train_dl: DataLoader, dev_dl: DataLoader,
          G: torch_ad.Function, H: torch_ad.Function):
    """Train the model on a given dataset."""

    optimiser = optim.AdamW(model.parameters(), lr=config.learning_rate, eps=1e-8)

    max_grad_norm = 1.0
    best_error = 0.
    loss_uk_values = []
    loss_k_values = []
    total_loss_values = []
    
    # Training loop
    for epoch_num in trange(config.epochs):
        logger.info(f"Epoch num: {epoch_num}")

        model.train()
        total_loss = 0.0
        total_loss_uk = 0.0
        total_loss_k = 0.0
        train_steps = len(train_dl)
        for step_num, batch in tqdm(enumerate(train_dl), total=train_steps):

            model.zero_grad()

            # Move batch to device
            batch = BatchedElement(*[x.to(config.device, non_blocking=True) if isinstance(x, torch.Tensor) else x for x in batch])
            k_exact = batch.target
            u_obs = batch.u_obs

            # Forward pass
            k = model(u_obs)

            # Solve PDE for κ_P and assemble the L2-loss: 0.5 * ||u(κ) - u_obs||^{2}_{L2}
            loss_uk = G(k, u_obs)
            total_loss_uk += loss_uk.item()

            # Assemble L2-loss: 0.5 * ||κ - κ_exact||^{2}_{L2}
            loss_k = H(k, k_exact)
            total_loss_k += loss_k.item()

            # Total loss
            loss = loss_k +  loss_uk / config.alpha
            total_loss += loss.item()

            # Backprop and perform Adam optimisation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            optimiser.step()

        
        logger.info(f"Total loss: {total_loss/train_steps}\
                    \n\t Loss u(κ): {total_loss_uk/train_steps/config.alpha}  Loss κ: {total_loss_k/train_steps}")

        loss_uk_values.append((total_loss_uk / (train_steps*config.alpha)))
        loss_k_values.append(total_loss_k / train_steps)
        total_loss_values.append(total_loss / train_steps)
        # Evaluation on dev set
        error = evaluate(model, config, dev_dl, disable_tqdm=True)
        logger.info(f"Error ({config.evaluation_metric}): {error}")

        # Save best-performing model
        if error < best_error or epoch_num == 0:
            best_error = error
            # Create directory for trained models
            name_dir = f"{config.dataset}-epoch-{epoch_num}-error_{best_error:.5f}"
            model_dir = os.path.join(config.data_dir, "saved_models", config.model_dir, name_dir)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            # Save model
            logger.info(f"Saving model checkpoint to {model_dir}\n")
            # Take care of distributed/parallel training
            model_to_save = (model.module if hasattr(model, "module") else model)
            torch.save(model_to_save.state_dict(), os.path.join(model_dir, "model.pt"))
            # Save training arguments together with the trained model
            config.to_file(os.path.join(model_dir, "training_args.json"))
    
    # plot_k = batch.target
    # plot_u = batch.u_obs
    # print("k,batch.target", plot_k)
    # print("batch.u_obs", plot_u)
    
    # xaxis = [i for i in range(config.epochs)]
    # plot image of loss

    plt.figure(figsize=(12, 6))
    plt.plot(loss_uk_values, label="Loss u(κ)")
    plt.plot(loss_k_values, label="Loss κ")
    plt.plot(total_loss_values, label="Total Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.title("Training Losses over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

    return model


if __name__ == "__main__":
    logger = get_logger("Training")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.environ["DATA_DIR"], type=str, help="Data directory")
    parser.add_argument("--model", default="cnn", type=str, help="one of [encoder-decoder, cnn]")
    parser.add_argument("--alpha", default=1e5, type=float, help="Regularisation parameter")
    parser.add_argument("--epochs", default=50, type=int, help="Epochs")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="Learning rate")
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout rate")
    parser.add_argument("--evaluation_metric", default="L2", type=str, help="Evaluation metric: one of [Lp, H1, Hdiv, Hcurl, , avg_rel]")
    parser.add_argument("--max_eval_steps", default=5000, type=int, help="Maximum number of evaluation steps")
    parser.add_argument("--dataset", default="heat_conductivity", type=str, help="Dataset name")
    parser.add_argument("--model_dir", default="model", type=str, help="Directory name to save trained models")
    parser.add_argument("--device", default="cpu", type=str, help="Device identifier (e.g. 'cuda:0' or 'cpu')")

    args = parser.parse_args()
    config = ModelConfig(**dict(args._get_kwargs()))

    # -- Load dataset -- #

    # Load train dataset
    train_dataset = PDEDataset(dataset=config.dataset, dataset_split="train", data_dir=config.data_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=train_dataset.collate, shuffle=False)
    # Load test dataset
    test_dataset = PDEDataset(dataset=config.dataset, dataset_split="test", data_dir=config.data_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, collate_fn=test_dataset.collate, shuffle=False)

    # -- Set PDE inputs (mesh, function space, boundary conditions, ...) -- #

    # Get mesh from dataset
    mesh = train_dataset.mesh
    # Define function space and test function
    V = FunctionSpace(mesh, "CG", 1)
    V_vec = VectorFunctionSpace(mesh, "CG", 1)  # create a vector function space
    x, y = SpatialCoordinate(V_vec.ufl_domain())
    # Apply a random force as f which like heat source in "heat"
    f_ramdon = as_vector([sin(pi * x) * sin(pi * y), sin(pi * x) * sin(pi * y)])
    f = interpolate(f_ramdon, V_vec)

    # -- Define the Firedrake operations to be composed with PyTorch -- #

    def solve_pde(k, u_obs, f, V):
        # us = 0
        # """Solve pde problem"""
        # v, u_ = TestFunction(V_vec), TrialFunction(V_vec)
        # u = Function(V_vec, name="Displacement")
        # E = Constant(1.0)  # Young's modulus
        # nu = Constant(0.2)

        # # Lamé parameter
        # lmbda = E*nu/(1+nu)/(1-2*nu)
        # mu = E/2/(1+nu)
        
        # # Define strain and stress's formular
        # def eps(v):
        #     return sym(grad(v))
        # def sig(v):
        #     d = 2
        #     return lmbda*tr(eps(v))*Identity(d) + 2*mu*eps(v)

        # # Apply random_field as input strain in x, y, xy direction
        # # Instead of store the tensor, I store the strain in each direction
        # # Expand the number of random fields to three times original N to ensure that 
        # # the number of strain fields generated is consistent with N.
        
        # # Define the displacement boundary conditions
        # bc_expr = np.array([k[0] + k[2] * x, k[1] + k[2] * y])
        # # Boundary conditions
        # bcL = DirichletBC(V_vec, bc_expr, 1)
        # bcR = DirichletBC(V_vec, bc_expr, 2)
        # bcB = DirichletBC(V_vec, bc_expr, 3)
        # bcT = DirichletBC(V_vec, bc_expr, 4)
        # # Define variational problem using the principle of virtual work
        # a = inner(sig(u_), eps(v)) * dx
        # L = inner(f, v) * dx

        # # Solve PDE
        # solve(a == L, u, bcs=[bcL, bcB, bcR, bcT], solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})

        # # Use the displacement to calculate the stress in each direction
        # sxx = sig(u)[0, 0]
        # syy = sig(u)[1, 1]
        # sxy = sig(u)[0, 1]

        # # Save the stress fields
        # sigma = []
        # sigma.append(sxx)
        # sigma.append(syy)
        # sigma.append(sxy)

        # us.append(sigma)
        u = u_obs
        # Assemble Firedrake L2-loss (and not l2-loss as in PyTorch)
        return assemble_L2_error(u, u_obs)

    def assemble_L2_error(x, x_exact):
        """Assemble L2-loss"""
        return assemble(0.5 * (x - x_exact) ** 2 * dx)

    solve_pde = functools.partial(solve_pde, f=f, V=V)

    # -- Construct the Firedrake torch operators -- #

    k = Function(V)
    u_obs = Function(V)
    k_exact = Function(V)

    # Set tape locally to only record the operations relevant to G on the computational graph
    with set_working_tape() as tape:
        # Define PyTorch operator for solving the PDE and compute the L2 error (for computing κ -> 0.5 * ||u(κ) - u_obs||^{2}_{L2})
        F = ReducedFunctional(solve_pde(k, u_obs), [Control(k), Control(u_obs)])
        G = torch_operator(F)

    # Set tape locally to only record the operations relevant to H on the computational graph
    with set_working_tape() as tape:
        # Define PyTorch operator for computing the L2-loss (for computing κ -> 0.5 * ||κ - κ_exact||^{2}_{L2})
        F = ReducedFunctional(assemble_L2_error(k, k_exact), [Control(k), Control(k_exact)])
        H = torch_operator(F)

    # -- Set the model -- #

    config.add_input_shape(V.dim())
    if config.model == "encoder-decoder":
        model = EncoderDecoder(config)
    elif config.model == "cnn":
        model = CNN(config)
    elif config.model == "resnet":
        model = ResNet(config)
    else:
        raise ValueError(f"Unknown model: {config.model}")

    # Set double precision (default Firedrake type)
    model.double()
    # Move model to device
    model.to(config.device)

    # -- Training -- #

    train(model, config=config, train_dl=train_dataloader, dev_dl=test_dataloader, G=G, H=H)

    # print(f"Dataset length: {len(train_dataset)}")
    # print(f"First data example: {train_dataset[0]}")
    # from firedrake import plot
    # import matplotlib.pyplot as plt

    # # 假设 `batch.u_obs` 是一个二维矢量场，我们可以选择一个分量进行绘制。
    # u_obs_firedrake = Function(V).assign(batch.u_obs)

    # # 选择一个分量进行绘制。
    # u_obs_component = u_obs_firedrake.sub(0)

    # fig, axes = plt.subplots()
    # axes.set_aspect('equal')
    # contourf = plot(u_obs_component, axes=axes)
    # fig.colorbar(contourf)
    # plt.title("Heatmap of u_obs component")
    # plt.show()

