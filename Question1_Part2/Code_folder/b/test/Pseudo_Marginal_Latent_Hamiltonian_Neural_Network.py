# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:44:42 2025

@author: EMRE
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Model definition
class ImprovedExtendedHNN(tf.keras.Model):
    def __init__(self, param_dim=2, aux_dim=2, hidden_dim=100):
        super().__init__()
        self.total_dim = param_dim + aux_dim

        # Network to learn H(θ, ρ, u, p)
        self.potential_net = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='tanh'),
            tf.keras.layers.Dense(hidden_dim, activation='tanh'),
            tf.keras.layers.Dense(1)  # Outputs potential energy
        ])

    def kinetic_energy(self, rho, u, p):
        """Compute kinetic energy term 1/2(ρᵀρ + uᵀu + pᵀp)"""
        return 0.5 * (tf.reduce_sum(rho**2, axis=1, keepdims=True) +
                     tf.reduce_sum(u**2, axis=1, keepdims=True) +
                     tf.reduce_sum(p**2, axis=1, keepdims=True))

    def call(self, inputs):
        """Compute full Hamiltonian"""
        theta, rho, u, p = inputs
        V = self.potential_net(tf.concat([theta, u], axis=1))
        T = self.kinetic_energy(rho, u, p)
        return V + T

    @tf.function
    def compute_gradients(self, theta, rho, u, p):
        """Compute gradients for equations of motion"""
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([theta, rho, u, p])
            H = self.call([theta, rho, u, p])

        dH_dtheta = tape.gradient(H, theta)
        dH_drho = tape.gradient(H, rho)
        dH_du = tape.gradient(H, u)
        dH_dp = tape.gradient(H, p)

        del tape
        return dH_dtheta, dH_drho, dH_du, dH_dp

def leapfrog_step(model, theta, rho, u, p, dt):
    """Single leapfrog integration step"""
    # Half step momentum
    dH_dtheta, _, dH_du, _ = model.compute_gradients(theta, rho, u, p)
    rho_half = rho - 0.5 * dt * dH_dtheta
    p_half = p - 0.5 * dt * (u + dH_du)

    # Full step position
    _, dH_drho, _, dH_dp = model.compute_gradients(theta, rho_half, u, p_half)
    theta_new = theta + dt * dH_drho
    u_new = u + dt * dH_dp

    # Half step momentum
    dH_dtheta, _, dH_du, _ = model.compute_gradients(theta_new, rho_half, u_new, p_half)
    rho_new = rho_half - 0.5 * dt * dH_dtheta
    p_new = p_half - 0.5 * dt * (u_new + dH_du)

    return theta_new, rho_new, u_new, p_new

# Data generation
def generate_pm_hmc_data(n_trajectories=100, n_steps=50, dt=0.1, param_dim=2, aux_dim=2):
    """Generate training data following PM-HMC dynamics"""
    def log_prior(theta):
        """Simple Gaussian prior: -log p(θ)"""
        return 0.5 * tf.reduce_sum(theta**2, axis=1, keepdims=True)

    def simulated_likelihood(theta, u):
        """Simple simulated likelihood: -log p̂(y|θ,u)"""
        return 0.5 * tf.reduce_sum((theta - u)**2, axis=1, keepdims=True)

    def hamiltonian_dynamics(theta, rho, u, p):
        """Equations of motion from equation 12"""
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([theta, u])
            V = log_prior(theta) + simulated_likelihood(theta, u)

        grad_theta = tape.gradient(V, theta)
        grad_u = tape.gradient(V, u)
        del tape

        dtheta_dt = rho
        drho_dt = -grad_theta
        du_dt = p
        dp_dt = -u - grad_u

        return dtheta_dt, drho_dt, du_dt, dp_dt

    # Generate trajectories
    all_states = []
    all_derivs = []

    for _ in range(n_trajectories):
        theta = tf.random.normal([1, param_dim])
        rho = tf.random.normal([1, param_dim])
        u = tf.random.normal([1, aux_dim])
        p = tf.random.normal([1, aux_dim])

        states = []
        derivs = []

        for _ in range(n_steps):
            states.append([theta, rho, u, p])
            dtheta, drho, du, dp = hamiltonian_dynamics(theta, rho, u, p)
            derivs.append([dtheta, drho, du, dp])

            # Leapfrog integration
            rho_half = rho + 0.5 * dt * drho
            theta = theta + dt * rho_half
            rho = rho_half + 0.5 * dt * drho

            p_half = p + 0.5 * dt * dp
            u = u + dt * p_half
            p = p_half + 0.5 * dt * dp

        all_states.extend(states)
        all_derivs.extend(derivs)

    all_states = [tf.concat(x, axis=0) for x in zip(*all_states)]
    all_derivs = [tf.concat(x, axis=0) for x in zip(*all_derivs)]

    return tf.data.Dataset.from_tensor_slices((
        tuple(all_states), tuple(all_derivs)
    )).shuffle(1000).batch(32)

def improved_train_step(model, states, derivs, optimizer):
    """Training step with improved energy conservation"""
    theta, rho, u, p = states
    true_dtheta, true_drho, true_du, true_dp = derivs

    with tf.GradientTape() as tape:
        # Dynamics loss
        dH_dtheta, dH_drho, dH_du, dH_dp = model.compute_gradients(theta, rho, u, p)
        dynamics_loss = (
            tf.reduce_mean(tf.square(dH_drho - true_dtheta)) +
            tf.reduce_mean(tf.square(-dH_dtheta - true_drho)) +
            tf.reduce_mean(tf.square(dH_dp - true_du)) +
            tf.reduce_mean(tf.square(-dH_du + u - true_dp))
        )

        # Energy conservation with increased weight
        H_start = model.call([theta, rho, u, p])
        theta_end, rho_end, u_end, p_end = leapfrog_step(model, theta, rho, u, p, dt=0.1)
        H_end = model.call([theta_end, rho_end, u_end, p_end])
        energy_loss = tf.reduce_mean(tf.square(H_end - H_start))

        # Total loss with higher energy conservation weight
        loss = dynamics_loss + 0.5 * energy_loss

    # Compute and clip gradients
    grads = tape.gradient(loss, model.trainable_variables)
    grads = [tf.clip_by_norm(g, 1.0) for g in grads]
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, dynamics_loss, energy_loss

def train_and_visualize():
    # Set random seed
    tf.random.set_seed(42)

    # Model parameters
    param_dim = 2
    aux_dim = 2
    n_epochs = 100

    # Create model and optimizer
    model = ImprovedExtendedHNN(param_dim=param_dim, aux_dim=aux_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # Generate training and test data
    train_data = generate_pm_hmc_data(n_trajectories=100)
    test_data = generate_pm_hmc_data(n_trajectories=20)

    # Training loop
    dynamics_losses = []
    energy_losses = []
    total_losses = []

    print("Starting training...")
    for epoch in range(n_epochs):
        epoch_dynamics_loss = 0
        epoch_energy_loss = 0
        epoch_total_loss = 0
        n_batches = 0

        for states, derivs in train_data:
            loss, d_loss, e_loss = improved_train_step(model, states, derivs, optimizer)
            epoch_total_loss += loss
            epoch_dynamics_loss += d_loss
            epoch_energy_loss += e_loss
            n_batches += 1

        total_losses.append(epoch_total_loss / n_batches)
        dynamics_losses.append(epoch_dynamics_loss / n_batches)
        energy_losses.append(epoch_energy_loss / n_batches)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}")
            print(f"Total Loss: {total_losses[-1]:.4f}")
            print(f"Dynamics Loss: {dynamics_losses[-1]:.4f}")
            print(f"Energy Loss: {energy_losses[-1]:.4f}\n")

    # Plot training history
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.plot(total_losses)
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')

    plt.subplot(132)
    plt.plot(dynamics_losses)
    plt.title('Dynamics Loss')
    plt.xlabel('Epoch')
    plt.yscale('log')

    plt.subplot(133)
    plt.plot(energy_losses)
    plt.title('Energy Conservation Loss')
    plt.xlabel('Epoch')
    plt.yscale('log')

    plt.tight_layout()
    plt.show()

    # Generate and plot trajectories
    for states, _ in test_data.take(1):
        theta, rho, u, p = states

        # Generate trajectory using leapfrog integration
        timesteps = 200
        dt = 0.05

        thetas = [theta.numpy()]
        rhos = [rho.numpy()]
        us = [u.numpy()]
        ps = [p.numpy()]
        hamiltonians = [model.call([theta, rho, u, p]).numpy()]

        for _ in range(timesteps):
            theta, rho, u, p = leapfrog_step(model, theta, rho, u, p, dt)

            thetas.append(theta.numpy())
            rhos.append(rho.numpy())
            us.append(u.numpy())
            ps.append(p.numpy())
            hamiltonians.append(model.call([theta, rho, u, p]).numpy())

        thetas = np.array(thetas)
        rhos = np.array(rhos)
        us = np.array(us)
        ps = np.array(ps)
        hamiltonians = np.array(hamiltonians)

        # Plot results
        plt.figure(figsize=(20, 5))

        plt.subplot(141)
        plt.plot(thetas[:, 0], rhos[:, 0])
        plt.xlabel('θ₁')
        plt.ylabel('ρ₁')
        plt.title('θ-ρ Phase Space')

        plt.subplot(142)
        plt.plot(us[:, 0], ps[:, 0])
        plt.xlabel('u₁')
        plt.ylabel('p₁')
        plt.title('u-p Phase Space')

        plt.subplot(143)
        plt.plot(range(len(hamiltonians)), hamiltonians[:, 0], 'g-')
        plt.xlabel('Time Step')
        plt.ylabel('H')
        plt.title('Hamiltonian Conservation')

        plt.subplot(144)
        for i in range(param_dim):
            plt.plot(range(len(thetas)), thetas[:, i], label=f'θ_{i+1}')
            plt.plot(range(len(rhos)), rhos[:, i], '--', label=f'ρ_{i+1}')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.title('Parameter Evolution')
        plt.legend()

        plt.tight_layout()
        plt.show()

    return model

if __name__ == "__main__":
    trained_model = train_and_visualize()

 