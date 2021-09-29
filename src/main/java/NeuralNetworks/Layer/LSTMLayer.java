package NeuralNetworks.Layer;

import NeuralNetworks.Neuron.LSTMNeuron;
import NeuralNetworks.Neuron.Neuron;

import java.io.Serializable;
import java.util.Random;
import Math.*;

public class LSTMLayer extends RecurrentLayer implements Serializable {

    private final Vector vector;
    private final Vector forgetVector;
    private final Vector addVector;
    private final Vector gVector;

    public LSTMLayer(int size, int nextSize, int seed, boolean isHidden) {
        super();
        this.size = size;
        this.vector = new Vector(size, 0);
        this.forgetVector = new Vector(size, 0);
        this.addVector = new Vector(size, 0);
        this.gVector = new Vector(size, 0);
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
        vector.clear();
        forgetVector.clear();
        addVector.clear();
        gVector.clear();
    }

    public Vector forgetVectorElementProduct(Vector vector) throws VectorSizeMismatch {
        return forgetVector.elementProduct(vector);
    }

    public Vector addVectorElementProduct() throws VectorSizeMismatch {
        return addVector.elementProduct(gVector);
    }

    public void addVector(int index, double value) {
        vector.addValue(index, value);
    }

    public double getVector(int index) {
        return vector.getValue(index);
    }

    public void setVector(int index, double value) {
        vector.setValue(index, value);
    }

    public void addForgetVector(int index, double value) {
        forgetVector.addValue(index, value);
    }

    public double getForgetVector(int index) {
        return forgetVector.getValue(index);
    }

    public void setForgetVector(int index, double value) {
        forgetVector.setValue(index, value);
    }

    public void addAddVector(int index, double value) {
        addVector.addValue(index, value);
    }

    public double getAddVector(int index) {
        return addVector.getValue(index);
    }

    public void setAddVector(int index, double value) {
        addVector.setValue(index, value);
    }

    public void addGVector(int index, double value) {
        gVector.addValue(index, value);
    }

    public double getGVector(int index) {
        return gVector.getValue(index);
    }

    public void setGVector(int index, double value) {
        gVector.setValue(index, value);
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
}
