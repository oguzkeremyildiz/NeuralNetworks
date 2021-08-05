import java.io.Serializable;

public class Neuron implements Serializable {

    private double value;
    private double[] weights;

    public Neuron() {
        value = 0.0;
        weights = null;
    }

    public Neuron(double[] weights) {
        this.value = 0.0;
        this.weights = weights;
    }

    public double getWeight(int i) {
        return weights[i];
    }

    public void setWeight(int i, double weight) {
        weights[i] = weight;
    }

    public void addWeight(int i, double weight) {
        weights[i] += weight;
    }

    public double getValue() {
        return value;
    }

    public void setValue(double value) {
        this.value = value;
    }

    public void initializeWeight(int size) {
        weights = new double[size];
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(value).append(" ");
        if (weights != null) {
            for (double weight : weights) {
                sb.append(weight).append(" ");
            }
        }
        return sb.toString();
    }
}