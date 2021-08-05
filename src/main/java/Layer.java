import java.io.Serializable;
import java.util.Random;

public class Layer implements Serializable {

    private final Neuron[] neurons;
    private final int size;
    private final int nextSize;

    public Layer(int size, int nextSize, int seed) {
        this.size = size;
        this.nextSize = nextSize;
        Random random = new Random(seed);
        this.neurons = new Neuron[size];
        for (int i = 0; i < neurons.length; i++) {
            double[] weights = new double[nextSize];
            for (int j = 0; j < weights.length; j++) {
                weights[j] = 2 * random.nextDouble() - 1;
            }
            neurons[i] = new Neuron(weights);
        }
    }

    public Layer(int size) {
        this.size = size;
        this.nextSize = 0;
        this.neurons = new Neuron[size];
        for (int i = 0; i < neurons.length; i++) {
            neurons[i] = new Neuron();
        }
    }

    public Neuron getNeuron(int i) {
        return neurons[i];
    }

    public void softmax() {
        double total = 0.0;
        for (int i = 0; i < size(); i++) {
            total += Math.exp(getNeuron(i).getValue());
        }
        for (int i = 0; i < size(); i++) {
            double value = getNeuron(i).getValue();
            getNeuron(i).setValue(Math.exp(value) / total);
        }
    }

    public double[][] neuronsToMatrix() {
        double[][] neurons = new double[1][this.neurons.length + 1];
        neurons[0][0] = 1.0;
        for (int i = 0; i < this.neurons.length; i++) {
            neurons[0][i + 1] = this.neurons[i].getValue();
        }
        return neurons;
    }

    public double[] neuronsToVector() {
        double[] neurons = new double[this.neurons.length];
        for (int i = 0; i < this.neurons.length; i++) {
            neurons[i] = this.neurons[i].getValue();
        }
        return neurons;
    }

    public double[][] weightsToMatrix() {
        double[][] weights = new double[size][nextSize];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < nextSize; j++) {
                weights[i][j] = neurons[i].getWeight(j);
            }
        }
        return weights;
    }

    public int size() {
        return size;
    }
}