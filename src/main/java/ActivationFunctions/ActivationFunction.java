package ActivationFunctions;

public interface ActivationFunction {
    double calculateForward(double value);
    double[][] calculateBack(double[] values);
}
