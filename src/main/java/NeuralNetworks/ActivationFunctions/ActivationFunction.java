package NeuralNetworks.ActivationFunctions;

import Math.*;
public interface ActivationFunction {
    double calculateForward(double value);
    Matrix calculateBack(Vector values);
}
