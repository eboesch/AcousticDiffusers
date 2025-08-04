import numpy as np
import pandas as pd


def compute_diffusion_coeffs(pressures):
    """
    Computes diffusion coefficients from pressure values.
    
    Args:
        pressures (np.ndarray): Array of pressure values. Shape: (n_frequencies, n_listeners).
        
    Returns:
        np.ndarray: Array of diffusion coefficients. Shape: (n_frequencies,).
    """
 
    L = pressures.shape[1] # Number of Listeners

    y = abs(pressures)**2 # shape: (n_frequencies, L)
    # const = 1e8
    # y *= const
    # y /= np.max(y, axis=1, keepdims=True) + 1e-20  # add epsilon to avoid /0

    sum_of_squares = np.sum(y**2, axis=1) # shape: (n_frequencies,)
    coeffs = (np.sum(y, axis=1)**2 - sum_of_squares)/( (L-1)*sum_of_squares) # shape: (n_frequencies,)


    # print("sum_y**2:", sum_y**2)
    print("sum_y_squared:", sum_of_squares)
    print("denominator:", (L - 1) * sum_of_squares)

    return coeffs


if __name__ == "__main__":

    image_path = "n01532829/n01532829_124.JPEG"
    sample = np.load('n01532829_124.npy')
    # sample = sample.reshape(sample.shape[0], 32,32,32)


    # TODO: fix the function! computed coefficients are WAY off
    coeffs = compute_diffusion_coeffs(sample)
    print(f"Computed Coefficients: {coeffs}")


    df = pd.read_csv("mini_imagenet/coeffs.csv")
    old_target = df.loc[df["path"]==image_path,:].drop(columns="path").to_numpy()[0]
    new_target = np.array([0.7407027313495752,0.7307803158049289,0.714483600722845,0.6943708440147114,0.6702065441524672,0.6442424410781448,0.6066966670054424,0.5175793434649474,0.397826593251609,0.34628255701429184,0.25898276914245133,0.20305107273700948,0.24770258574501872,0.17388251406248828,0.2331128118245641,0.2542189398845884,0.17169546205053512])
    # print(f"Old target: {old_target}")
    # print(f"New target: {new_target}")

    print(f"Difference: {new_target-coeffs}")

