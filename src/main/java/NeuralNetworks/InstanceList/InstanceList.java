package NeuralNetworks.InstanceList;

import NeuralNetworks.*;

import java.io.Serializable;
import java.util.*;

public class InstanceList extends BasicInstanceList<String> implements Serializable {

    private HashMap<String, Integer>[] maps = null;

    public InstanceList(Scanner source, String regex, AttributeType attributesType) throws InputSizeException {
        super();
        int currentKey = -1;
        while (source.hasNext()) {
            String[] line = source.nextLine().split(regex);
            if (input < 0) {
                input = line.length - 1;
                if (attributesType.equals(AttributeType.DISCRETE)) {
                    maps = new HashMap[input];
                    for (int i = 0; i < maps.length; i++) {
                        maps[i] = new HashMap<>();
                    }
                }
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
            if (!reverseClasses.containsKey(currentKey)) {
                reverseClasses.put(currentKey, classInfo);
            }
            list.add(new Instance<>());
            for (int i = 0; i < line.length - 1; i++) {
                String feature = line[i];
                if (attributesType.equals(AttributeType.DISCRETE)) {
                    if (!maps[i].containsKey(feature)) {
                        maps[i].put(feature, maps[i].size());
                    }
                }
                list.get(list.size() - 1).add(feature);
            }
            list.get(list.size() - 1).add(line[line.length - 1]);
        }
    }

    public int inputSize() {
        if (maps == null) {
            return -1;
        }
        int count = 0;
        for (HashMap<String, Integer> map : maps) {
            count += map.size();
        }
        return count;
    }

    public int mapSize(int index) {
        return maps[index].size();
    }

    public int getFeature(int index, String feature) {
        if (!maps[index].containsKey(feature)) {
            return -1;
        }
        return maps[index].get(feature);
    }
}
