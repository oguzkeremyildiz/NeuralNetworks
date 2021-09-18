package NeuralNetworks.Layer;

import java.io.Serializable;
import java.util.Random;
import Math.*;
import NeuralNetworks.Neuron.Neuron;
import NeuralNetworks.Neuron.RecurrentNeuron;

public class RecurrentLayer extends Layer implements Serializable {

    public RecurrentLayer() {
    }

    public RecurrentLayer(int size, int nextSize, int seed) {
        super();
        this.size = size;
        this.nextSize = nextSize;
        Random random = new Random(seed);
        this.neurons = new Neuron[size];
        for (int i = 0; i < neurons.length; i++) {
            double[] weights = new double[nextSize];
            double[] recurrentWeights = new double[size];
            for (int j = 0; j < weights.length; j++) {
                weights[j] = 2 * random.nextDouble() - 1;
            }
            for (int j = 0; j < recurrentWeights.length; j++) {
                recurrentWeights[j] = 2 * random.nextDouble() - 1;
            }
            neurons[i] = new RecurrentNeuron(weights, recurrentWeights);
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

    public Matrix oldNeuronsToMatrix() {
        Matrix neurons = new Matrix(1, this.neurons.length);
        for (int i = 0; i < this.neurons.length; i++) {
            neurons.setValue(0, i, ((RecurrentNeuron)this.neurons[i]).getOldValue());
        }
        return neurons;
    }
}
