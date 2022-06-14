package NeuralNetworks.Neuron;

import NeuralNetworks.Initializer.InitializerFunction;

import java.io.Serializable;
import java.util.Random;

public class LSTMBias extends Bias implements Serializable {

    private final double[][] gateWeights;

    public LSTMBias(int seed, int size, InitializerFunction function) {
        super(seed, size, function);
        Random random = new Random(seed);
        this.gateWeights = new double[3][size];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < size; j++) {
                gateWeights[i][j] = function.calculate(random);
            }
        }
    }

    public double getGateValue(int gate, int i) {
        return gateWeights[gate][i];
    }

    public void addGateValue(int gate, int index, double weight) {
        gateWeights[gate][index] += weight;
    }
}
