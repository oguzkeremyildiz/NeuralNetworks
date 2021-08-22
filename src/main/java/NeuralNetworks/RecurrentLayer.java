package NeuralNetworks;

import java.io.Serializable;
import java.util.Random;

public class RecurrentLayer extends Layer implements Serializable {

    public RecurrentLayer(int size, int nextSize, int seed) {
        super();
        this.size = size;
        this.nextSize = nextSize;
        Random random = new Random(seed);
        this.neurons = new Neuron[size];
        for (int i = 0; i < neurons.length; i++) {
            double[] weights = new double[nextSize];
            for (int j = 0; j < weights.length; j++) {
                weights[j] = 2 * random.nextDouble() - 1;
            }
            neurons[i] = new RecurrentNeuron(weights, size);
        }
    }

    public void setValuesToZero() {
        for (Neuron neuron : neurons) {
            ((RecurrentNeuron) neuron).setOldValue(0);
        }
    }

    public void setValues() {
        for (Neuron neuron : neurons) {
            ((RecurrentNeuron) neuron).setOldValue();
        }
    }
}
