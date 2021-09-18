package NeuralNetworks.InstanceList;

import java.io.Serializable;
import java.util.LinkedList;
import java.util.Scanner;

import Dictionary.VectorizedDictionary;
import Dictionary.VectorizedWord;
import Math.Vector;
import NeuralNetworks.InputSizeException;

public class VectorizedInstanceList extends BasicInstanceList<java.util.Vector<String>> implements Serializable {

    private java.util.Vector<String> convert(Vector vec) {
        java.util.Vector<String> result = new java.util.Vector<>();
        for (int i = 0; i < vec.size(); i++) {
            result.add(Double.toString(vec.getValue(i)));
        }
        return result;
    }

    public VectorizedInstanceList(Scanner source, VectorizedDictionary dictionary, LinkedList<AttributeType> attributeTypes, String regex) throws InputSizeException {
        super();
        int layerSize = -1;
        if (dictionary != null) {
            layerSize = dictionary.mostSimilarWord(dictionary.getWord(0).toString()).getVector().size();
        }
        int classSize = -1;
        while (source.hasNext()) {
            String[] array = source.nextLine().split(regex);
            this.list.add(new Instance<>());
            for (int i = 0; i < array.length; i += attributeTypes.size() + 1) {
                java.util.Vector<String> vec = new java.util.Vector<>();
                for (int j = i; j < i + attributeTypes.size(); j++) {
                    switch (attributeTypes.get(j - i)) {
                        case BINARY:
                            if ("true".equals(array[j])) {
                                vec.add("1");
                            } else {
                                vec.add("0");
                            }
                            break;
                        case DISCRETE:
                            VectorizedWord vectorizedWord = dictionary.mostSimilarWord(array[j]);
                            if (vectorizedWord != null) {
                                vec = convert(vectorizedWord.getVector());
                            } else {
                                for (int k = 0; k < layerSize; k++) {
                                    vec.add("0");
                                }
                            }
                            break;
                        case CONTINUOUS:
                            vec.add(array[j]);
                            break;
                    }
                }
                String classInfo = array[i + attributeTypes.size()];
                if (!this.classes.containsKey(classInfo)) {
                    classSize++;
                    this.classes.put(classInfo, classSize);
                }
                if (!this.reverseClasses.containsKey(classSize)) {
                    this.reverseClasses.put(classSize, classInfo);
                }
                if (this.input == -1) {
                    this.input = vec.size();
                } else {
                    if (vec.size() != input) {
                        throw new InputSizeException();
                    }
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
