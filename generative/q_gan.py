# Import necessary libraries

# Import the os module to interact with the operating system.
import os

# Define q_gan function
def q_gan(df, X, X_train, X_test, y, y_train, y_test, reps=None, ibm_account=None, quantum_backend=None, output_folder=None, data_columns=None):

    # Import necessary libraries
    import os
    import torch
    import pandas as pd
    import numpy as np
    from sklearn import preprocessing
    from qiskit.utils import algorithm_globals
    from qiskit_machine_learning.datasets.dataset_helper import discretize_and_truncate
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from qiskit import IBMQ, Aer, QuantumCircuit
    from qiskit.circuit.library import TwoLocal
    from qiskit_machine_learning.connectors import TorchConnector
    from qiskit_machine_learning.neural_networks import CircuitQNN
    from torch.optim import Adam
    import torch.nn as nn
    import torch.nn.functional as F
    from qiskit.opflow import Gradient, StateFn
    from IPython.display import clear_output
    from scipy.stats import entropy
    from matplotlib import cm
    from qiskit.utils import QuantumInstance

    # Import necessary libraries
    from sklearn import preprocessing
    from sklearn.neighbors import NeighborhoodComponentsAnalysis

    # Data rescaling with RobustScaler
    # Instantiate a RobustScaler object which is robust to outliers
    scaler = preprocessing.RobustScaler()

    # Fit the scaler to the feature data (X) and transform it
    # The transformed data will have the same number of features as the input,
    # but with a different scale
    X_train = scaler.fit_transform(X_train)

    # Load and apply Neighborhood Components Analysis (NCA)
    # Instantiate a NeighborhoodComponentsAnalysis object with n_components=2,
    # using PCA for initialization and a fixed random state for reproducibility
    extraction = NeighborhoodComponentsAnalysis(n_components=2, init="pca", random_state=42)

    # Fit the NCA model to the feature data (X) and the target variable (y)
    # and transform the data using the learned model
    # The transformed data (training_data) will have 2 components, as specified
    # by n_components
    training_data = extraction.fit(X_train, y_train).transform(X_train)

    # Display the transformed training_data
    print(training_data)

    # Fixing seeds in the random number generators
    torch.manual_seed(42)
    algorithm_globals.random_seed = 42

    #training_data = X_train.to_numpy()
    #print("Training Data for qGAN")
    #print(training_data)

    # Define minimal and maximal values for the training data
    bounds_min = np.percentile(training_data, 5, axis=0)
    bounds_max = np.percentile(training_data, 95, axis=0)

    # Create a list of bounds for each dimension of the training data
    bounds = []
    for i, _ in enumerate(bounds_min):
        bounds.append([bounds_min[i], bounds_max[i]])

    # Determine data resolution for each dimension of the training data in terms of the number of qubits used to represent each data dimension
    data_dim = [3, 3]

    # Pre-processing, i.e., discretization of the data (gridding)
    (training_data, grid_data, grid_elements, prob_data) = discretize_and_truncate(
        training_data,
        np.asarray(bounds),
        data_dim,
        return_data_grid_elements=True,
        return_prob=True,
        prob_non_zero=True,
    )

    # Display the bounds
    print("Display the bounds")
    print(bounds)

    # Convert training_data and grid_elements to PyTorch tensors with a float data type
    training_data = torch.tensor(training_data, dtype=torch.float)
    grid_elements = torch.tensor(grid_elements, dtype=torch.float)

    # Define the training batch size
    batch_size = 10

    # Create a DataLoader object to handle batching of the training_data
    dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last=True)

    # Create a histogram of the first and second variables of the training_data using matplotlib
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    ax1.hist(training_data[:, 0], bins=20)
    ax1.set_title("Histogram of the first variable")
    ax1.set_xlabel("Values")
    ax1.set_ylabel("Counts")
    ax2.hist(training_data[:, 1], bins=20)
    ax2.set_title("Histogram of the second variable")
    ax2.set_xlabel("Values")
    ax2.set_ylabel("Counts")
    save = output_folder + 'histogram_q_gan.png'
    fig.savefig(save)

    # Save and load the IBMQ account with the provided API key
    IBMQ.save_account(ibm_account, overwrite=True)
    IBMQ.load_account()

    # Get the provider and the backend for the quantum simulations
    provider = IBMQ.get_provider(hub='ibm-q')
    backend = provider.get_backend(quantum_backend)

    # Create QuantumInstance objects for training and sampling, setting the number of shots
    qi_training = QuantumInstance(backend, shots=batch_size)
    qi_sampling = QuantumInstance(backend, shots=1024)

    # Assuming data_dim is already defined as a list of data resolutions for each dimension

    # Create a quantum circuit (qc) with a total number of qubits equal to the sum of data_dim
    qc = QuantumCircuit(sum(data_dim))

    # Apply Hadamard (H) gates to all qubits in the quantum circuit
    qc.h(qc.qubits)

    # Choose a hardware-efficient ansatz
    twolocal = TwoLocal(sum(data_dim), "ry", "cx", reps=2, entanglement="sca")

    # Compose the quantum circuit (qc) with the TwoLocal object (twolocal)
    qc.compose(twolocal, inplace=True)

    # Draw the decomposed quantum circuit using the matplotlib (mpl) backend
    qc.decompose().draw("mpl")

    # Function to create a generator using a TorchConnector and a CircuitQNN
    def create_generator(quantum_instance) -> TorchConnector:
        circuit_qnn = CircuitQNN(
            qc,
            input_params=[],
            weight_params=qc.parameters,
            quantum_instance=quantum_instance,
            sampling=True,
            sparse=False,
            interpret=lambda x: grid_elements[x],
        )

        return TorchConnector(circuit_qnn)

    # Define a Discriminator class as a subclass of nn.Module
    class Discriminator(nn.Module):
        def __init__(self, input_size):
            super(Discriminator, self).__init__()

            self.linear_input = nn.Linear(input_size, 20)
            self.leaky_relu = nn.LeakyReLU(0.2)
            self.linear20 = nn.Linear(20, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            x = self.linear_input(input)
            x = self.leaky_relu(x)
            x = self.linear20(x)
            x = self.sigmoid(x)
            return x

    # Define generator and discriminator loss functions
    gen_loss_fun = nn.BCELoss()
    disc_loss_fun = nn.BCELoss()

    # Define the generator gradient
    generator_grad = Gradient().gradient_wrapper(
        StateFn(qc), twolocal.ordered_parameters, backend=qi_training
    )
    
    # Define the generator loss gradient function
    def generator_loss_grad(parameter_values, discriminator):
        # Evaluate gradient
        grads = generator_grad(parameter_values).tolist()

        # Initialize the list to store loss gradients
        loss_grad_list = []

        # Iterate through the gradients
        for j, grad in enumerate(grads):
            cx = grad[0].tocoo()
            input = torch.zeros(len(cx.col), len(data_dim))
            target = torch.ones(len(cx.col), 1)
            weight = torch.zeros(len(cx.col), 1)

            # Update input and weight tensors
            for i, (index, prob_grad) in enumerate(zip(cx.col, cx.data)):
                input[i, :] = grid_elements[index]
                weight[i, :] = prob_grad

            # Calculate Binary Cross Entropy loss gradient
            bce_loss_grad = F.binary_cross_entropy(discriminator(input), target, weight)
            loss_grad_list.append(bce_loss_grad)

        # Stack the list of loss gradients into a single tensor
        loss_grad = torch.stack(loss_grad_list)
        return loss_grad


    # Define a function to calculate the relative entropy between generated data and true data
    def get_relative_entropy(gen_data) -> float:
        prob_gen = np.zeros(len(grid_elements))

        # Calculate the probability of generated data
        for j, item in enumerate(grid_elements):
            for gen_item in gen_data.detach().numpy():
                if np.allclose(np.round(gen_item, 6), np.round(item, 6), rtol=1e-5):
                    prob_gen[j] += 1

        # Normalize the probability
        prob_gen = prob_gen / len(gen_data)
        prob_gen = [1e-8 if x == 0 else x for x in prob_gen]

        # Return the relative entropy
        return entropy(prob_gen, prob_data)


    # Initialize generator and discriminator
    generator = create_generator(qi_training)
    discriminator = Discriminator(len(data_dim))

    # Define hyperparameters
    lr = 0.01  # learning rate
    b1 = 0.9  # first momentum parameter
    b2 = 0.999  # second momentum parameter
    num_epochs = 100  # number of training epochs

    # Optimizer for the generator
    optimizer_gen = Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    # Optimizer for the discriminator
    optimizer_disc = Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    # Function to plot training progress
    def plot_training_progress():
        # We don't plot if we don't have enough data
        if len(generator_loss_values) < 2:
            return

        # Clear previous output
        clear_output(wait=True)

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

        # Plot loss on the first subplot
        ax1.set_title("Loss")
        ax1.plot(generator_loss_values, label="generator loss", color="royalblue")
        ax1.plot(discriminator_loss_values, label="discriminator loss", color="magenta")
        ax1.legend(loc="best")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Loss")
        ax1.grid()

        # Plot relative entropy on the second subplot
        ax2.set_title("Relative entropy")
        ax2.plot(relative_entropy_values)
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Relative entropy")
        ax2.grid()

        # Save the figure to a file
        save_path = output_folder + 'Loss_Relative_Entropy_q_gan.png'
        fig.savefig(save_path)

    # Initialize lists for storing relative entropy, generator loss, and discriminator loss values
    relative_entropy_values = []
    generator_loss_values = []
    discriminator_loss_values = []

    # Training loop
    for epoch in range(num_epochs):
        # Initialize lists for storing relative entropy, generator loss, and discriminator loss values for the current epoch
        relative_entropy_epoch = []
        generator_loss_epoch = []
        discriminator_loss_epoch = []

        # Iterate over the batches of data points
        for i, data in enumerate(dataloader):
            # Set adversarial ground truths for real and fake data points
            valid = torch.ones(data.size(0), 1)
            fake = torch.zeros(data.size(0), 1)

            # Generate a batch of data points using the generator
            gen_data = generator()

            # Evaluate relative entropy for the generated data points
            relative_entropy_epoch.append(get_relative_entropy(gen_data))

            # Train the discriminator
            optimizer_disc.zero_grad()

            # Compute discriminator loss based on its ability to distinguish real from generated samples
            disc_data = discriminator(data)
            real_loss = disc_loss_fun(disc_data, valid)
            fake_loss = disc_loss_fun(discriminator(gen_data), fake)
            discriminator_loss = (real_loss + fake_loss) / 2

            # Perform backpropagation on discriminator loss and update discriminator parameters
            discriminator_loss.backward(retain_graph=True)
            optimizer_disc.step()

            # Train the generator
            optimizer_gen.zero_grad()

            # Compute generator loss based on its ability to prepare good data samples
            generator_loss = gen_loss_fun(discriminator(gen_data), valid)
            generator_loss.retain_grad = True
            g_loss_grad = generator_loss_grad(generator.weight.data.numpy(), discriminator)

            # Assign gradient values to generator parameters and update generator parameters
            for j, param in enumerate(generator.parameters()):
                param.grad = g_loss_grad
            optimizer_gen.step()

            # Store generator and discriminator losses for the current batch
            generator_loss_epoch.append(generator_loss.item())
            discriminator_loss_epoch.append(discriminator_loss.item())

        # Compute and store average relative entropy, generator loss, and discriminator loss values for the current epoch
        relative_entropy_values.append(np.mean(relative_entropy_epoch))
        generator_loss_values.append(np.mean(generator_loss_epoch))
        discriminator_loss_values.append(np.mean(discriminator_loss_epoch))

    # Plot the training progress
    plot_training_progress()

    # Create a generator for sampling using the optimized weights
    generator_sampling = create_generator(qi_sampling)
    generator_sampling.weight.data = generator.weight.data

    # Generate data using the created generator
    gen_data = generator_sampling().detach().numpy()
    prob_gen = np.zeros(len(grid_elements))

    # Calculate the probability distribution for generated data
    for j, item in enumerate(grid_elements):
        for gen_item in gen_data:
            if np.allclose(np.round(gen_item, 6), np.round(item, 6), rtol=1e-5):
                prob_gen[j] += 1
    prob_gen = prob_gen / len(gen_data)

    # Replace zero probabilities with a very small value (1e-8)
    prob_gen = [1e-8 if x == 0 else x for x in prob_gen]

    # Plot the cumulative distribution function for generated and training data
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(111, projection="3d")
    ax1.set_title("Cumulative Distribution Function")

    # Plot generated data CDF
    ax1.bar3d(
        np.transpose(grid_elements)[1],
        np.transpose(grid_elements)[0],
        np.zeros(len(prob_gen)),
        0.05,
        0.05,
        np.cumsum(prob_gen),
        label="generated data",
        color="blue",
        alpha=1,
    )

    # Plot training data CDF
    ax1.bar3d(
        np.transpose(grid_elements)[1] + 0.05,
        np.transpose(grid_elements)[0] + 0.05,
        np.zeros(len(prob_data)),
        0.05,
        0.05,
        np.cumsum(prob_data),
        label="training data",
        color="orange",
        alpha=1,
    )

    # Set axis labels
    ax1.set_xlabel("x_1")
    ax1.set_ylabel("x_0")
    ax1.set_zlabel("p(x)")

    # Save the figure
    save = output_folder + 'Cumulative_Distribution_Function_q_gan.png'
    fig.savefig(save)

    # Apply the inverse PCA transformation to the generated data
    X_orig = np.dot(gen_data, extraction.components_)

    # Inverse scale the data to its original range
    X_orig_backscaled = scaler.inverse_transform(X_orig)

    # Define the column names for the DataFrame
    column_names = list(data_columns)

    # Create a DataFrame with the back-scaled data and the column names
    data = pd.DataFrame(X_orig_backscaled, columns=data_columns)
    data_X_test = pd.DataFrame(X_test, columns=data_columns)
    data_X = pd.DataFrame(X, columns=data_columns)
        
    # Save the DataFrame to a CSV file
    output_file = output_folder + 'data_gan_generated.csv'
    output_file_X_test = output_folder + 'data_gan_X_test.csv'
    output_file_X = output_folder + 'data_gan_X.csv'
    data.to_csv(output_file, index=False)
    data_X_test.to_csv(output_file_X_test, index=False)
    data_X.to_csv(output_file_X, index=False)
