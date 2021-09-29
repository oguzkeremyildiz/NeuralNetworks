package NeuralNetworks.Neuron;

import java.io.Serializable;
import java.util.Random;

public class LSTMBias extends Bias implements Serializable {

    private final double[] forgetGateWeights;
    private final double[] addGateWeights;
    private final double[] gGateWeights;

    public LSTMBias(int seed, int size) {
        super(seed, size);
        Random random = new Random(seed);
        forgetGateWeights = new double[size];
        addGateWeights = new double[size];
        gGateWeights = new double[size];
        for (int i = 0; i < size; i++) {
            forgetGateWeights[i] = 2 * random.nextDouble() - 1;
            addGateWeights[i] = 2 * random.nextDouble() - 1;
            gGateWeights[i] = 2 * random.nextDouble() - 1;
        }
    }

    public double getForgetGateValue(int i) {
        return forgetGateWeights[i];
    }

    public void addForgetGateWeight(int index, double weight) {
        this.forgetGateWeights[index] += weight;
    }

    public double getAddGateValue(int i) {
        return addGateWeights[i];
    }

    public void addAddGateWeight(int index, double weight) {
        this.addGateWeights[index] += weight;
    }

    public double getGGateValue(int i) {
        return gGateWeights[i];
    }

    public void addGGateWeight(int index, double weight) {
        this.gGateWeights[index] += weight;
    }
}
