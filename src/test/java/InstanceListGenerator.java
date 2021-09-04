import Corpus.Corpus;
import Dictionary.VectorizedDictionary;
import NeuralNetworks.AttributeType;
import NeuralNetworks.InputSizeException;
import NeuralNetworks.InstanceList.VectorizedInstanceList;
import WordToVec.NeuralNetwork;
import WordToVec.WordToVecParameter;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.LinkedList;
import java.util.Scanner;
import Math.*;

public class InstanceListGenerator {

    public static void main(String[] args) throws FileNotFoundException, MatrixColumnMismatch, VectorSizeMismatch, InputSizeException {
        NeuralNetwork neuralNetwork = new NeuralNetwork(new Corpus("src/main/resources/Dataset/NER/dictionary.txt"), new WordToVecParameter());
        VectorizedDictionary dictionary = neuralNetwork.train();
        LinkedList<AttributeType> attributeTypes = new LinkedList<>();
        attributeTypes.add(AttributeType.DISCRETE);
        attributeTypes.add(AttributeType.CONTINUOUS);
        attributeTypes.add(AttributeType.CONTINUOUS);
        attributeTypes.add(AttributeType.CONTINUOUS);
        attributeTypes.add(AttributeType.CONTINUOUS);
        attributeTypes.add(AttributeType.CONTINUOUS);
        attributeTypes.add(AttributeType.CONTINUOUS);
        attributeTypes.add(AttributeType.CONTINUOUS);
        attributeTypes.add(AttributeType.CONTINUOUS);
        attributeTypes.add(AttributeType.CONTINUOUS);
        attributeTypes.add(AttributeType.CONTINUOUS);
        attributeTypes.add(AttributeType.CONTINUOUS);
        attributeTypes.add(AttributeType.CONTINUOUS);
        attributeTypes.add(AttributeType.CONTINUOUS);
        attributeTypes.add(AttributeType.CONTINUOUS);
        attributeTypes.add(AttributeType.CONTINUOUS);
        attributeTypes.add(AttributeType.CONTINUOUS);
        attributeTypes.add(AttributeType.CONTINUOUS);
        attributeTypes.add(AttributeType.CONTINUOUS);
        attributeTypes.add(AttributeType.CONTINUOUS);
        attributeTypes.add(AttributeType.CONTINUOUS);
        attributeTypes.add(AttributeType.CONTINUOUS);
        attributeTypes.add(AttributeType.CONTINUOUS);
        attributeTypes.add(AttributeType.CONTINUOUS);
        attributeTypes.add(AttributeType.CONTINUOUS);
        attributeTypes.add(AttributeType.CONTINUOUS);
        attributeTypes.add(AttributeType.CONTINUOUS);
        attributeTypes.add(AttributeType.CONTINUOUS);
        attributeTypes.add(AttributeType.CONTINUOUS);
        attributeTypes.add(AttributeType.CONTINUOUS);
        attributeTypes.add(AttributeType.CONTINUOUS);
        attributeTypes.add(AttributeType.CONTINUOUS);
        attributeTypes.add(AttributeType.CONTINUOUS);
        VectorizedInstanceList list = new VectorizedInstanceList(new Scanner(new File("src/main/resources/Dataset/SentimentAnalysis/data-w-f-p-train.txt")), dictionary, attributeTypes, " ");
        list.save("src/main/resources/Dataset/SentimentAnalysis/sentiment-list-2-train.bin");
        list = new VectorizedInstanceList(new Scanner(new File("src/main/resources/Dataset/SentimentAnalysis/data-w-f-p-test.txt")), dictionary, attributeTypes, " ");
        list.save("src/main/resources/Dataset/SentimentAnalysis/sentiment-list-2-test.bin");
    }
}
