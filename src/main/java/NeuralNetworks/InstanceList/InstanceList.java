package NeuralNetworks.InstanceList;

import NeuralNetworks.*;

import java.io.Serializable;
import java.util.*;

public class InstanceList extends BasicInstanceList<String> implements Serializable {

    public InstanceList(Scanner source, String regex) throws InputSizeException {
        super();
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
            if (!reverseClasses.containsKey(currentKey)) {
                reverseClasses.put(currentKey, classInfo);
            }
            list.add(new Instance<>());
            for (String s : line) {
                list.get(list.size() - 1).add(s);
            }
        }
    }
}
