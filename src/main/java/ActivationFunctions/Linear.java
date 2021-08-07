package ActivationFunctions;

import java.io.Serializable;

public class Linear implements ActivationFunction, Serializable {
    @Override
    public double calculateForward(double value) {
        return value;
    }

    @Override
    public double[][] calculateBack(double[] values) {
        double[][] vector = new double[values.length][1];
        for (int i = 0; i < values.length; i++) {
            vector[i][0] = 1.0;
        }
        return vector;
    }

    @Override
    public String toString() {
        return "LINEAR";
    }
}
