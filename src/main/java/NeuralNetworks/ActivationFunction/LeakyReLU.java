package NeuralNetworks.ActivationFunction;

import java.io.Serializable;
import Math.*;

public class LeakyReLU implements ActivationFunction, Serializable {
    @Override
    public double calculateForward(double value) {
        return Math.max(0.1 * value, value);
    }

    @Override
    public Matrix calculateBack(Vector values) {
        Matrix matrix = new Matrix(values.size(), 1);
        for (int i = 0; i < values.size(); i++) {
            matrix.setValue(i, 0, 0.1);
            if (values.getValue(i) > 0) {
                matrix.setValue(i, 0, 1.0);
            }
        }
        return matrix;
    }

    @Override
    public String toString() {
        return "LEAKYRELU";
    }
}
