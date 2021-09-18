package NeuralNetworks.ActivationFunction;

import java.io.Serializable;
import Math.*;

public class TanH implements ActivationFunction, Serializable {
    @Override
    public double calculateForward(double value) {
        return (2.0D / (1.0D + Math.exp(-2 * value))) - 1.0;
    }

    @Override
    public Matrix calculateBack(Vector values) {
        Matrix matrix = new Matrix(values.size(), 1);
        for (int i = 0; i < values.size(); i++) {
            matrix.setValue(i, 0, 1.0 - (values.getValue(i) * values.getValue(i)));
        }
        return matrix;
    }

    @Override
    public String toString() {
        return "TANH";
    }
}
