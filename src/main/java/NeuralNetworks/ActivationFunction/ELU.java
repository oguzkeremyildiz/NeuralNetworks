package NeuralNetworks.ActivationFunction;

import java.io.Serializable;
import Math.*;

public class ELU implements ActivationFunction, Serializable {

    @Override
    public double calculateForward(double value) {
        if (value > 0) {
            return value;
        }
        return Math.exp(value) - 1;
    }

    @Override
    public Matrix calculateBack(Vector values) {
        Matrix matrix = new Matrix(values.size(), 1);
        for (int i = 0; i < values.size(); i++) {
            double value = values.getValue(i);
            if (value > 0) {
                matrix.setValue(i, 0, 1);
            } else {
                matrix.setValue(i, 0, value + 1);
            }
        }
        return matrix;
    }

    @Override
    public String toString() {
        return "ELU";
    }
}
