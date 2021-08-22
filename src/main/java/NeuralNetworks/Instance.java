package NeuralNetworks;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;

public class Instance<T> implements Serializable {

    private final ArrayList<T> list;

    public Instance() {
        this.list = new ArrayList<>();
    }

    @SafeVarargs
    public Instance(T... elements) {
        this.list = new ArrayList<>();
        list.addAll(Arrays.asList(elements));
    }

    public void add(T s) {
        list.add(s);
    }

    public T get(int index) {
        return list.get(index);
    }

    public T getLast() {
        return list.get(list.size() - 1);
    }

    public int size() {
        return list.size();
    }
}
