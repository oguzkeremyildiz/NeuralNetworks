package NeuralNetworks.Neuron;

import java.io.Serializable;

public class RecurrentNeuron extends Neuron implements Serializable {

    private final double[] recurrentWeights;
    private double oldValue;

    public RecurrentNeuron(double[] weights, double[] recurrentWeights) {
        super(weights);
        oldValue = 0.0;
        this.recurrentWeights = recurrentWeights;
    }

    public RecurrentNeuron(double[] weights) {
        super(weights);
        oldValue = Integer.MIN_VALUE;
        this.recurrentWeights = null;
    }

    public double getRecurrentWeight(int i) {
        return recurrentWeights[i];
    }

    public void addRecurrentWeight(int index, double value) {
        recurrentWeights[index] += value;
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
