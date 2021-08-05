import java.io.Serializable;
import java.util.*;

public class InstanceList implements Serializable {

    private ArrayList<Instance> list;
    private HashMap<String, Integer> classes;
    private int input = -1;

    public InstanceList(Scanner source, String regex) throws InputSizeException {
        this.list = new ArrayList<>();
        this.classes = new HashMap<>();
        int currentKey = -1;
        while (source.hasNext()) {
            String[] line = source.nextLine().split(regex);
            if (input < 0) {
                input = line.length - 1;
            } else {
                if (line.length - 1 != input) {
                    throw new InputSizeException();
                }
            }
            String classInfo = line[line.length - 1];
            if (!classes.containsKey(classInfo)) {
                currentKey++;
                classes.put(classInfo, currentKey);
            }
            list.add(new Instance());
            for (String s : line) {
                list.get(list.size() - 1).add(s);
            }
        }
    }

    public Instance getInstance(int i) {
        return list.get(i);
    }

    public void shuffle(int seed) {
        Collections.shuffle(list, new Random(seed));
    }

    public int size() {
        return list.size();
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

    public int get(String key) {
        return classes.get(key);
    }

    public String get(int neuron) {
        for (String key : classes.keySet()) {
            if (classes.get(key) == neuron) {
                return key;
            }
        }
        return null;
    }
}
