package NeuralNetworks.Layer;

import NeuralNetworks.Neuron.LSTMNeuron;
import NeuralNetworks.Neuron.Neuron;

import java.io.Serializable;
import java.util.Random;

public class LSTMLayer extends RecurrentLayer implements Serializable {

    public LSTMLayer(int size, int nextSize, int seed, boolean isHidden) {
        super();
        this.size = size;
        this.nextSize = nextSize;
        Random random = new Random(seed);
        this.neurons = new Neuron[size];
        for (int i = 0; i < neurons.length; i++) {
            double[] weights = new double[nextSize];
            double[] forgetWeights = new double[nextSize];
            double[] addWeights = new double[nextSize];
            double[] gWeights = new double[nextSize];
            for (int j = 0; j < nextSize; j++) {
                weights[j] = 2 * random.nextDouble() - 1;
                forgetWeights[j] = 2 * random.nextDouble() - 1;
                addWeights[j] = 2 * random.nextDouble() - 1;
                gWeights[j] = 2 * random.nextDouble() - 1;
            }
            if (isHidden) {
                double[] recurrentWeights = new double[size];
                double[] forgetRecurrentWeights = new double[size];
                double[] addRecurrentWeights = new double[size];
                double[] gRecurrentWeights = new double[size];
                for (int j = 0; j < size; j++) {
                    recurrentWeights[j] = 2 * random.nextDouble() - 1;
                    forgetRecurrentWeights[j] = 2 * random.nextDouble() - 1;
                    addRecurrentWeights[j] = 2 * random.nextDouble() - 1;
                    gRecurrentWeights[j] = 2 * random.nextDouble() - 1;
                }
                neurons[i] = new LSTMNeuron(weights, recurrentWeights, forgetWeights, forgetRecurrentWeights, addWeights, addRecurrentWeights, gWeights, gRecurrentWeights);
            } else {
                neurons[i] = new LSTMNeuron(weights, forgetWeights, addWeights, gWeights);
            }
        }
    }
}
