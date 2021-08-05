package ActivationFunctions;

import java.io.Serializable;

public class Sigmoid implements ActivationFunction, Serializable {

    @Override
    public double calculateForward(double value) {
        return 1.0D / (1.0D + Math.exp(-(Double)value));
    }

    private double[][] multiply(double[] values, double[] oneMinusValues) {
        double[][] vector = new double[values.length][1];
        for (int i = 0; i < values.length; i++) {
            vector[i][0] = values[i] * oneMinusValues[i];
        }
        return vector;
    }

    @Override
    public double[][] calculateBack(double[] values) {
        double[] oneMinusValues = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            oneMinusValues[i] = 1.0 - values[i];
        }
        return multiply(values, oneMinusValues);
    }

    @Override
    public String toString() {
        return "SIGMOID";
    }
}
