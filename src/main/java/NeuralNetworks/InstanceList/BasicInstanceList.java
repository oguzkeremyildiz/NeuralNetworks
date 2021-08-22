package NeuralNetworks.InstanceList;

import NeuralNetworks.Instance;

import java.io.Serializable;
import java.util.*;

public class BasicInstanceList<T> implements Serializable {

    protected final ArrayList<Instance<T>> list;
    protected final HashMap<String, Integer> classes;
    protected final HashMap<Integer, String> reverseClasses;
    protected int input = -1;

    public BasicInstanceList() {
        this.classes = new HashMap<>();
        this.reverseClasses = new HashMap<>();
        this.list = new ArrayList<>();
    }

    public int get(String key) {
        return classes.get(key);
    }

    public String get(int neuron) {
        return reverseClasses.get(neuron);
    }

    public int getInput() {
        return input;
    }

    public int getOutput() {
        if (classes.size() == 2) {
            return 1;
        }
        return classes.size();
    }

    public Instance<T> getInstance(int i) {
        return list.get(i);
    }

    public void shuffle(int seed) {
        Collections.shuffle(list, new Random(seed));
    }

    public int size() {
        return list.size();
    }
}