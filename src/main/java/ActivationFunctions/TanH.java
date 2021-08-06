package ActivationFunctions;

import java.io.Serializable;

public class TanH implements ActivationFunction, Serializable {
    @Override
    public double calculateForward(double value) {
        return (2.0D / (1.0D + Math.exp(-2 * value))) - 1.0;
    }

    @Override
    public double[][] calculateBack(double[] values) {
        double[][] vector = new double[values.length][1];
        for (int i = 0; i < values.length; i++) {
            vector[i][0] = 1.0 - (values[i] * values[i]);
        }
        return vector;
    }

    @Override
    public String toString() {
        return "TANH";
    }
}
