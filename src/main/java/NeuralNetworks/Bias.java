package NeuralNetworks;

import java.io.Serializable;
import java.util.Random;

public class Bias implements Serializable {

    private final double[] weights;

    public Bias(int seed, int size) {
        Random random = new Random(seed);
        weights = new double[size];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = 2 * random.nextDouble() - 1;
        }
    }

    public Bias(int size) {
        this.weights = new double[size];
    }

    public double getValue(int i) {
        return weights[i];
    }

    public void setWeight(int index, double weight) {
        this.weights[index] = weight;
    }

    public void addWeight(int index, double weight) {
        this.weights[index] += weight;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (double weight : weights) {
            sb.append(weight).append(" ");
        }
        return sb.toString();
    }
}
