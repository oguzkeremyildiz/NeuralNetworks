package NeuralNetworks;

import java.io.Serializable;
import java.util.Arrays;

public class RecurrentNeuron extends Neuron implements Serializable {

    private double[] recurrentWeights;
    private double oldValue;

    public RecurrentNeuron(double[] weights, int recurrentWeightsSize) {
        super(weights);
        oldValue = 0.0;
        recurrentWeights = new double[recurrentWeightsSize];
        Arrays.fill(recurrentWeights, 0.0);
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
