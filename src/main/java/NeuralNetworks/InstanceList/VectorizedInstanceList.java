package NeuralNetworks.InstanceList;

import java.io.Serializable;
import java.util.LinkedList;
import java.util.Scanner;

import Dictionary.VectorizedDictionary;
import Math.Vector;
import NeuralNetworks.Instance;

public class VectorizedInstanceList extends BasicInstanceList<java.util.Vector<String>> implements Serializable {

    private java.util.Vector<String> convert(Vector vec) {
        java.util.Vector<String> result = new java.util.Vector<>();
        for (int i = 0; i < vec.size(); i++) {
            result.add(Double.toString(vec.getValue(i)));
        }
        return result;
    }

    public VectorizedInstanceList(Scanner source, VectorizedDictionary dictionary, int featureSize, String regex) {
        super();
        int classSize = -1;
        while (source.hasNext()) {
            String[] array = source.nextLine().split(regex);
            this.list.add(new Instance<>());
            for (int i = 0; i < array.length; i += featureSize + 2) {
                java.util.Vector<String> vec = convert(dictionary.mostSimilarWord(array[i]).getVector());
                for (int j = i + 1; j < i + featureSize + 1; j++) {
                    vec.add(array[j]);
                }
                String classInfo = array[i + featureSize + 1];
                if (!this.classes.containsKey(classInfo)) {
                    classSize++;
                    this.classes.put(classInfo, classSize);
                }
                if (!this.reverseClasses.containsKey(classSize)) {
                    this.reverseClasses.put(classSize, classInfo);
                }
                if (this.input == -1) {
                    this.input = vec.size();
                }
                this.list.get(this.list.size() - 1).add(vec);
                java.util.Vector<String> classVector = new java.util.Vector<>();
                classVector.add(classInfo);
                this.list.get(this.list.size() - 1).add(classVector);
            }
        }
    }

    public LinkedList<String> collectClassInfos(Instance<java.util.Vector<String>> instance) {
        LinkedList<String> classes = new LinkedList<>();
        for (int i = 1; i < instance.size(); i += 2) {
            classes.add(instance.get(i).get(0));
        }
        return classes;
    }
}
