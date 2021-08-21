package NeuralNetworks.ActivationFunctions;

import java.io.Serializable;
import Math.*;

public class Sigmoid implements ActivationFunction, Serializable {

    @Override
    public double calculateForward(double value) {
        return 1.0D / (1.0D + Math.exp(-(Double)value));
    }

    private Matrix multiply(Vector values, Vector oneMinusValues) {
        Matrix matrix = new Matrix(values.size(), 1);
        for (int i = 0; i < values.size(); i++) {
            matrix.setValue(i, 0, values.getValue(i) * oneMinusValues.getValue(i));
        }
        return matrix;
    }

    @Override
    public Matrix calculateBack(Vector values) {
        Vector oneMinusValues = new Vector(values.size(), 0);
        for (int i = 0; i < values.size(); i++) {
            oneMinusValues.setValue(i, 1.0 - values.getValue(i));
        }
        return multiply(values, oneMinusValues);
    }

    @Override
    public String toString() {
        return "SIGMOID";
    }
}
