import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def read_spectrum_file_numpy(file_path):
    metadata = {}
    data = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                key_value = line[1:].strip().split(None, 1)
                if len(key_value) == 2:
                    metadata[key_value[0]] = key_value[1]
            else:
                try:
                    wavelength, flux = map(float, line.split(","))
                    data.append((wavelength, flux))
                except ValueError:
                    print(f"Skipping invalid data line: {line}")

    data_array = np.array(data)
    return metadata, data_array

def linear(x, a, b):
    return a * x + b

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

def plot_spectrum(file_path):
    metadata, spectrum_data = read_spectrum_file_numpy(file_path)
    wavelengths = spectrum_data[:, 0]
    fluxes = spectrum_data[:, 1]

    # Create a scatter plot for the original data
    plt.subplot(3, 1, 1)
    plt.scatter(wavelengths, fluxes, c='blue', s=10)
    plt.title('Spectrum Data: Wavelength vs Flux')
    plt.xlabel('Wavelength (Angstrom)')
    plt.ylabel('Flux (ADU)')

def fit_continuum_and_plot(file_path):
    metadata, spectrum_data = read_spectrum_file_numpy(file_path)
    wavelengths = spectrum_data[:, 0]
    fluxes = spectrum_data[:, 1]

    # Perform linear fit
    initial_guess = [0, np.mean(fluxes)]
    popt, _ = curve_fit(linear, wavelengths, fluxes, p0=initial_guess)
    fitted_continuum_line = linear(wavelengths, *popt)

    # Plot the original data and the fitted continuum
    plt.subplot(3, 1, 2)
    plt.scatter(wavelengths, fluxes, c='blue', s=10, label='Original Data')
    plt.plot(wavelengths, fitted_continuum_line, c='red', lw=2, label='Fitted Continuum Line')
    plt.title('Continuum Fit to Spectrum Data')
    plt.xlabel('Wavelength (Angstrom)')
    plt.ylabel('Flux (ADU)')
    plt.legend()

def fit_gaussian_continuum_and_plot(file_path):
    metadata, spectrum_data = read_spectrum_file_numpy(file_path)
    wavelengths = spectrum_data[:, 0]
    fluxes = spectrum_data[:, 1]

    # Fit the continuum
    popt_continuum, _ = curve_fit(linear, wavelengths, fluxes)
    continuum_fit = linear(wavelengths, *popt_continuum)
    fluxes_detrended = fluxes - continuum_fit

    # Fit the Gaussian
    initial_guess_gaussian = [np.max(fluxes_detrended), np.median(wavelengths), 10]
    popt_gaussian, _ = curve_fit(gaussian, wavelengths, fluxes_detrended, p0=initial_guess_gaussian)
    combined_fit = linear(wavelengths, *popt_continuum) + gaussian(wavelengths, *popt_gaussian)

    # Plot original data, fitted continuum, and Gaussian
    plt.subplot(3, 1, 3)
    plt.scatter(wavelengths, fluxes, c='blue', s=10, label='Original Data')
    plt.plot(wavelengths, continuum_fit, 'g--', lw=2, label='Fitted Continuum')
    plt.plot(wavelengths, combined_fit, c='red', lw=2, label='Fitted Gaussian + Continuum')
    plt.title('Gaussian Fit to Continuum Line')
    plt.xlabel('Wavelength (Angstrom)')
    plt.ylabel('Flux (ADU)')
    plt.legend()

# Usage
file_path = r'C:\Users\surya\Documents\GitHub\python-workshop-assignment-starscream3301/spectrum.txt'  # Use the correct file path

plt.figure(figsize=(10, 18))  # Adjust the figure size as needed
plot_spectrum(file_path)
fit_continuum_and_plot(file_path)
fit_gaussian_continuum_and_plot(file_path)

plt.tight_layout()  # Adjusts the layout to prevent overlap
plt.show()
