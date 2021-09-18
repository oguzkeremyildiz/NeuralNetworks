package NeuralNetworks.ActivationFunction;

import Math.*;
public interface ActivationFunction {
    double calculateForward(double value);
    Matrix calculateBack(Vector values);
}
