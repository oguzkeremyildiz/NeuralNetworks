import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;

public class Instance implements Serializable {

    private final ArrayList<String> list;

    public Instance() {
        this.list = new ArrayList<>();
    }

    public Instance(String... array) {
        this.list = new ArrayList<>();
        list.addAll(Arrays.asList(array));
    }

    public void add(String s) {
        list.add(s);
    }

    public String get(int index) {
        return list.get(index);
    }

    public String getLast() {
        return list.get(list.size() - 1);
    }
}
