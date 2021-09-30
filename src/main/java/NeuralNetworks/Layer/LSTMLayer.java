package NeuralNetworks.Layer;

import NeuralNetworks.Neuron.LSTMNeuron;
import NeuralNetworks.Neuron.Neuron;

import java.io.Serializable;
import java.util.LinkedList;
import java.util.Random;
import Math.*;

public class LSTMLayer extends RecurrentLayer implements Serializable {

    private final LinkedList<Vector> vectors;

    public LSTMLayer(int size, int nextSize, int seed, boolean isHidden) {
        super();
        this.size = size;
        this.vectors = new LinkedList<>();
        for (int i = 0; i < 4; i++) {
            vectors.add(new Vector(size, 0));
        }
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

    public void setContextValuesToZero() {
        for (Neuron neuron : this.neurons) {
            ((LSTMNeuron) neuron).setOldContextValue(0);
            ((LSTMNeuron) neuron).setContextValue(0);
        }
    }

    public void setContextValues() {
        for (Neuron neuron : this.neurons) {
            ((LSTMNeuron) neuron).setOldContextValue();
        }
    }

    public Vector oldContextValuesToVector() {
        Vector vector = new Vector(this.neurons.length, 0);
        for (int i = 0; i < this.neurons.length; i++) {
            Neuron neuron = this.neurons[i];
            vector.addValue(i, ((LSTMNeuron) neuron).getOldContextValue());
        }
        return vector;
    }

    public void clear() {
        for (Vector vector : this.vectors) {
            vector.clear();
        }
    }

    public Vector getVector(int index) {
        return this.vectors.get(index);
    }

    public Matrix forgetGateWeightsToMatrix() {
        Matrix weights = new Matrix(size, nextSize);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < nextSize; j++) {
                weights.setValue(i, j, ((LSTMNeuron) neurons[i]).getForgetGateWeight(j));
            }
        }
        return weights;
    }

    public Matrix addGateWeightsToMatrix() {
        Matrix weights = new Matrix(size, nextSize);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < nextSize; j++) {
                weights.setValue(i, j, ((LSTMNeuron) neurons[i]).getAddGateWeight(j));
            }
        }
        return weights;
    }

    public Matrix gGateWeightsToMatrix() {
        Matrix weights = new Matrix(size, nextSize);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < nextSize; j++) {
                weights.setValue(i, j, ((LSTMNeuron) neurons[i]).getGGateWeight(j));
            }
        }
        return weights;
    }
}
