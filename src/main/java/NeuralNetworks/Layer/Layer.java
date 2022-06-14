package NeuralNetworks.Layer;

import java.io.Serializable;
import java.util.LinkedList;
import java.util.Random;
import Math.*;
import NeuralNetworks.Initializer.InitializerFunction;

public class Layer implements Serializable {

    protected double[] values;
    protected double[][] weights;
    protected int size;
    protected int nextSize;

    public Layer(int s, LinkedList<Integer> hiddenLayers, int seed, InitializerFunction function) {
        this.size = hiddenLayers.get(s);
        this.nextSize = hiddenLayers.get(s + 1);
        Random random = new Random(seed);
        this.weights = new double[size][nextSize];
        this.values = new double[size];
        for (int i = 0; i < weights.length; i++) {
            for (int k = 0; k < weights[i].length; k++) {
                weights[i][k] = function.calculate(random);
            }
        }
    }

    public Layer(int size) {
        this.size = size;
        this.nextSize = 0;
        this.values = new double[size];
    }

    public double getWeight(int i, int j) {
        return weights[i][j];
    }

    public void addWeight(int i, int j, double weight) {
        weights[i][j] += weight;
    }

    public double getValue(int i) {
        return values[i];
    }

    public void setValue(int i, double value) {
        values[i] = value;
    }

    public void softmax() {
        double total = 0.0;
        for (int i = 0; i < size(); i++) {
            total += Math.exp(values[i]);
        }
        for (int i = 0; i < size(); i++) {
            double value = values[i];
            values[i] = Math.exp(value) / total;
        }
    }

    public Matrix neuronsToMatrix() {
        Matrix neurons = new Matrix(1, size + 1);
        neurons.setValue(0, 0, 1.0);
        for (int i = 0; i < size; i++) {
            neurons.setValue(0, i + 1, values[i]);
        }
        return neurons;
    }

    public Vector neuronsToVector() {
        Vector neurons = new Vector(size, 0);
        for (int i = 0; i < size; i++) {
            neurons.setValue(i, values[i]);
        }
        return neurons;
    }

    public Matrix weightsToMatrix() {
        Matrix weights = new Matrix(size, nextSize);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < nextSize; j++) {
                weights.setValue(i, j, getWeight(i, j));
            }
        }
        return weights;
    }

    public int size() {
        return size;
    }
}