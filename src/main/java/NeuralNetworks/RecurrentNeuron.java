package NeuralNetworks;

import java.io.Serializable;
import java.util.Random;

public class RecurrentNeuron extends Neuron implements Serializable {

    private double[] recurrentWeights;
    private double oldValue;

    public RecurrentNeuron(double[] weights, int recurrentWeightsSize, Random random) {
        super(weights);
        oldValue = 0.0;
        recurrentWeights = new double[recurrentWeightsSize];
        for (int i = 0; i < recurrentWeights.length; i++) {
            recurrentWeights[i] = 2 * random.nextDouble() - 1;
        }
    }

    public double getRecurrentWeight(int i) {
        return recurrentWeights[i];
    }

    public double setRecurrentWeight(int index, double value) {
        return recurrentWeights[index] = value;
    }

    public double getOldValue() {
        return oldValue;
    }

    public void setOldValue(double oldValue) {
        this.oldValue = oldValue;
    }

    public void setOldValue() {
        this.oldValue = this.value;
    }
}
