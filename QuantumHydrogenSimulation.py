# h2_hamiltonian_pipeline.py
import numpy as np
import pennylane as qml
from pennylane import qchem
import csv
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib import cm

# ---------- Core pipeline functions ------------
def get_hamiltonian_for_h2(bond_length):
    symbols = ["H", "H"]
    coords = np.array([[0.0, 0.0, 0.0],
                       [0.0, 0.0, bond_length]])
    H, n_qubits = qchem.molecular_hamiltonian(symbols, coords)
    return H, n_qubits


def hamiltonian_matrix_and_energies(H):
    # sparse_hamiltonian returns a sparse matrix
    sparse_H = H.sparse_matrix()
    H_dense = sparse_H.toarray()
    eigvals, eigvecs = np.linalg.eigh(H_dense)
    # eigvals sorted ascending, ground state is eigvals[0]
    return H_dense, eigvals, eigvecs

def make_energy_qnode(H, n_qubits):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def energy_for_state(statevector):
        qml.StatePrep(statevector, wires=range(n_qubits))
        return qml.expval(H)

    return energy_for_state

def make_time_evolution_qnode(H, n_qubits):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def evolve(t):
        qml.BasisState(np.array([1] + [0]*(n_qubits-1)), wires=range(n_qubits))
        qml.ApproxTimeEvolution(H, t, 1)
        return qml.expval(H)

    return evolve


def plot_3d_geometries(bond_lengths, time_idx=0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for R in bond_lengths:
        H1 = np.array([0,0,0])
        H2 = np.array([0,0,R])

        xs = [H1[0], H2[0]]
        ys = [H1[1], H2[1]]
        zs = [H1[2], H2[2]]

        ax.plot(xs, ys, zs, marker='o')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

# _______________________________________________________
# BROWNIAN MOTION SECTION
#--------------------------------------------------------

# Step 1: Initialize the quantum device
num_qubits = 2
dev = qml.device("default.qubit", wires=num_qubits)

# Step 2: Normalize data for amplitude encoding
def normalize_and_pad_data(data):
    # Normalize the data
    norm = np.linalg.norm(data)
    if norm == 0:
        normalized_data = data  # Avoid division by zero
    else:
        normalized_data = data / norm

    # Ensure the length matches 2^num_qubits (padding not needed as rows match 8 qubits)
    return normalized_data

# Step 3: Quantum circuit for amplitude encoding
@qml.qnode(dev)
def amplitude_encoding_circuit(data):
    """
    Quantum circuit to load data using amplitude encoding.
    - data: Normalized data to encode into quantum state amplitudes.
    """
    qml.AmplitudeEmbedding(features=data, wires=range(num_qubits), pad_with=0, normalize=False)
    return qml.state()

# Step 4: Perform the quantum random walk
def quantum_random_walk(steps, initial_value, lower_bound, upper_bound):
    """
    Simulates a quantum random walk for the given steps.
    - steps: Number of steps to project.
    - initial_value: Last value of the input dataset to start projections.
    - lower_bound: Minimum constraint for the forecasted values.
    - upper_bound: Maximum constraint for the forecasted values.
    """
    projections = [initial_value]
    angles = [0]
    weights = np.array([-0.05, -0.01, 0.01, 0.05])  # Example weights for the 4 quantum states

    for step in range(steps):
        # Generate a random angle for the rotation
        random_angle = np.random.uniform(0, 2 * np.pi) # Want to save random angles for visualization as well !!!!!!
        
        # Execute the quantum circuit and get probabilities
        probabilities = random_walk_circuit(random_angle)
        
        # Derive a new value using weighted probabilities
        new_value = projections[-1] + (np.dot(probabilities, weights))
        
        # Apply constraints to keep the value within bounds
        new_value = np.clip(new_value, lower_bound, upper_bound)
        projections.append(new_value)
        angles.append(random_angle)
    
    return [projections[1:],angles[1:]]

@qml.qnode(dev)
def random_walk_circuit(random_angle):
    """
    Quantum circuit for a single step of the quantum random walk with randomness.
   
    """
    # Apply random rotation to the qubit
    qml.RY(random_angle, wires=0)

    # Conditional shift operation
    qml.ctrl(qml.PauliX, control=0)(wires=1)

    # Measure probabilities
    return qml.probs(wires=[0, 1])

#______________________________________
# ANIMATION CREATION
#--------------------------------------

def animate_bond_evolution(bond_lengths, time_points, xyz,bond_energies):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    norm = Normalize(vmin=0, vmax=2*len(bond_energies))
    cmap = plt.cm.plasma

    # --- Set background to black ---
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # --- White labels, ticks, and spines ---
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')

    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')


    # Initialize atoms
    h1 = ax.scatter([0], [0], [0], s=100, color='black')
    h2 = ax.scatter([0], [0], [bond_lengths[0]], s=100, color='black')
    bond_line = Line3D([0, 0], [0, 0], [0, bond_lengths[0]], color="white", linewidth=3)
    ax.add_line(bond_line)

    ax.set_xlim(-xyz[0], xyz[0])
    ax.set_ylim(-xyz[1], xyz[1])
    ax.set_zlim(0, xyz[2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.title.set_color('white')

    # Update function for animation
    def update(frame):
        r = random.randrange(1,4)*0.001 #to add more visual variation (just to show the change more effectively)
        b = bond_lengths[frame]+r
        e = bond_energies[frame]
        color = cmap((bond_energies[frame])**2)
        h2._offsets3d = ([0], [0], [b])
        h1.set_color(color)
        h2.set_color(color)
        bond_line.set_data([0, 0], [0, 0])  
        bond_line.set_3d_properties([0, b])
        bond_line.set_color(color)
        ax.set_title(f"Bond length = {b:.4f} Å Energy = {e:.4f} Eh (Hartrees) (Step = {time_points[frame]:})")
        return h1, h2, bond_line

    ani = FuncAnimation(fig, update, frames=len(time_points), interval=80, blit=False)
    plt.show()
    return ani

def plotcompare(time, bonds,energies):
    fig, ax1 = plt.subplots(figsize=(10,5))

    # First axis = bond lengths
    ax1.set_xlabel("Time", color="white")
    ax1.set_ylabel("Bond Length (Å) (Blue)", color="white")
    ax1.plot(time, bonds, color="cyan", label="Bond Length", linewidth=2)
    ax1.tick_params(axis='both', colors="white")

    # Second axis = energies (shares the same x-axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Energy (Purple)", color="white")
    ax2.plot(time, energies, color="magenta", label="Energy", linewidth=2)
    ax2.tick_params(axis='both', colors="white")

    # Title + backgrounds
    fig.patch.set_facecolor("black")
    ax1.set_facecolor("black")
    ax2.set_facecolor("black")
    plt.title("Bond Length & Energy vs Time", color="white")

    # Optional: add grids
    ax1.grid(alpha=0.3)

    plt.show()


# Pipeline to run code
if __name__ == "__main__":
    bond_lengths = [0.5, 0.74, 1.0, 1.2]
    bond_lengths_over_time = np.array([[0.5,0.502,0.491,0.504],[0.74,0.741,0.737,0.735],[1.0,1.09,1.03,0.99],[1.2,1.21,1.22,1.17]])
    brownian_values = 4*[0]
    brownian_angles = 4*[0]
    num_steps = 100
    for i in range(0,4):
        data = bond_lengths_over_time[i]
        normalized_data = normalize_and_pad_data(data) 
        encoded_state = amplitude_encoding_circuit(normalized_data)
        brownian_values[i], brownian_angles[i] = quantum_random_walk(num_steps, data[3], min(data), max(data))
    # print(brownian_values,brownian_angles) # debugging
    results = []
    bond_energies = []
    time_points = np.linspace(0, 1.0, num_steps)
    for b in range(4): # fix this, we need to loop through the brownian_values for each bond length
        H, n_qubits = get_hamiltonian_for_h2(bond_lengths[b])
        H_mat, eigvals, eigvecs = hamiltonian_matrix_and_energies(H)

        ground_energy = eigvals[0]
        excited_energies = eigvals[1:]

        # Create QNodes for this Hamiltonian
        energy_qnode = make_energy_qnode(H, n_qubits)
        # evolve_qnode = make_time_evolution_qnode(H, n_qubits) maybe dont need this since geometry is using it

        # Prepare ground stateclassically from eigenvector and compute energy via QNode
        ground_state_vec = eigvecs[:, 0]  # eigenvector corresponding to lowest eigenvalue
        #print(ground_state_vec)
        measured_energy = energy_qnode(ground_state_vec)  # should match ground_energy

        # Do a time evolution starting from ground state and measure final state
        # To start evolution from ground state we need to prepare ground state first, but
        # our evolve_qnode doesn't support state initialization — we can make a new QNode that prepares the state then evolves
        dev = qml.device("default.qubit", wires=n_qubits)
        @qml.qnode(dev)
        def prepare_then_evolve_and_measure(time=0.5, steps=2, state_vec=ground_state_vec): 
            qml.StatePrep(state_vec, wires=range(n_qubits))
            qml.ApproxTimeEvolution(H, time, steps)
            return qml.expval(H)
        energies = num_steps*[0]
        #state_vec = ground_state_vec
        for i in range(num_steps):
            H, n_qubits = get_hamiltonian_for_h2(brownian_values[b][i])
            H_mat, eigvals, eigvecs = hamiltonian_matrix_and_energies(H)
            ground_state_vec = eigvecs[:, 0]
            state_vec = ground_state_vec
            energies[i] = float(prepare_then_evolve_and_measure(time_points[i], 2,state_vec))
        bond_energies.append(energies)
        results.append({
            "bond": b,
            "n_qubits": n_qubits,
            "ground_energy_diag": float(ground_energy),
            "measured_energy_qnode": float(measured_energy),
            # "energy_after_evolution": float(energy_after_evolution)
        })

    # Print results
    for r in results:
        print(f"bond {r['bond']:.3f} Å: groundE (diag) {r['ground_energy_diag']:.6f}, "
              f"measured {r['measured_energy_qnode']:.6f}")
    #for i in range(len(bond_energies)):
        #print(bond_energies[i]) # debugging

    # PLOTS and ANIMATIONS

    # plot_3d_geometries(bond_lengths) # initial plots

    timepoints = np.linspace(0, num_steps-1, num_steps)
    animate_bond_evolution(brownian_values[0],timepoints,[0.5,0.5,0.5],bond_energies[0])
    animate_bond_evolution(brownian_values[1],timepoints,[0.75,0.75,0.75],bond_energies[1])
    animate_bond_evolution(brownian_values[2],timepoints,[1,1,1],bond_energies[2])
    animate_bond_evolution(brownian_values[3],timepoints,[1.3,1.3,1.3],bond_energies[3])
    plotcompare(timepoints, brownian_values[0],bond_energies[0])
    plotcompare(timepoints, brownian_values[1],bond_energies[1])
    plotcompare(timepoints, brownian_values[2],bond_energies[2])
    plotcompare(timepoints, brownian_values[3],bond_energies[3])

