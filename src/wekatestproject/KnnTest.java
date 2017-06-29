/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekatestproject;

/**
 *
 * @author Avinash
 */

import weka.filters.unsupervised.attribute.StringToNominal;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.lazy.IBk;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.supervised.attribute.ClassConditionalProbabilities;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.MergeTwoValues;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.Reorder;
import weka.filters.unsupervised.attribute.SortLabels;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author tirathshah
 */

public class KnnTest {
    
    // Function to read file data
    public static BufferedReader readDataFile(String filename) {
        BufferedReader inputReader = null;

        try {
            inputReader = new BufferedReader(new FileReader(filename));
        } catch (FileNotFoundException ex) {
            System.err.println("File not found: " + filename);
        }

        return inputReader;
    }

    // Function to split data into training-testing dataset for the given number of folds
    public static Instances[][] crossValidationSplit(Instances data, int numberOfFolds) {
        Instances[][] split = new Instances[2][numberOfFolds];
        
        int count = 0;
        
        while (count < numberOfFolds) {
            
            split[0][count] = data.trainCV(numberOfFolds, count);
            split[1][count] = data.testCV(numberOfFolds, count);
            
            count += 1;
        }
        
        return split;
    }

    // Function to build classifier for training-testing set and return the validation
    public static Evaluation simpleClassify(Classifier model, Instances trainingSet, Instances testingSet) throws Exception {
        Evaluation validation = new Evaluation(trainingSet);

        model.buildClassifier(trainingSet);
        validation.evaluateModel(model, testingSet);
        
        return validation;
    }

    // Function to calculate accuracy of the prediction data
    public static double calculateAccuracy(ArrayList predictions, Instances data) {
        
        double accuracy = 0;
        int predictionSize = predictions.size();
        
        double num_key[] = data.attributeToDoubleArray(0);
        
        System.out.println("0: Good and 1: Bad");
        System.out.println("stmt_date \t rm_key \t expected \t predicted");
        
        int count = 0;
        
        while (count < predictionSize) {
            
            NominalPrediction nominalPrediction = (NominalPrediction) predictions.get(count);
            
            if (nominalPrediction.predicted() == nominalPrediction.actual()) {
                accuracy++;
            } else {
                System.out.println(data.get(count).toString(1) + "\t  " + (int)num_key[count] + "\t\t  " + (int)nominalPrediction.actual() + "   \t\t   " + (int)nominalPrediction.predicted());
            }
            count += 1;
        }
        
        System.out.println("\nTotal: " + (int)predictions.size() + "\tCorrectly Classified: " + (int)accuracy  + "\tIncorrect: " + (int)(predictions.size()-accuracy));
        
        return 100 * accuracy / predictionSize; // returns accuracy for the given prediction
    }

    // Function to convert into nominal attributes/features
    public static Instances discretizedata(Instances data_old){
        Instances data_new =null;
        
        try {
            Discretize discretize = new Discretize();
            discretize.setAttributeIndices("12-last");
            discretize.setInputFormat(data_old);
            data_new = Filter.useFilter(data_old, discretize);
        } catch (Exception ex) {
            Logger.getLogger(KnnTest.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        return data_new; // returns nominal data
    }

    // function to remove attributes/feautres from dataset
    public static Instances removeAtt(Instances data_old){

        // remove of attributes in a dataset
        Remove remove =new Remove();
        remove.setAttributeIndices("23,24,28,34,40");
        //remove.setInputFormat(data_old);
        Instances data_new=null;

        try {
            remove.setInputFormat(data_old);
            data_new = Filter.useFilter(data_old, remove);
        } catch (Exception ex) {
            Logger.getLogger(KnnTest.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        return data_new; // return the new data
    }

    // function to Merge A,B,C to one and Merge D,F to one
    public static Instances mergeValues(Instances data_old){

        // creating Instances for merging and classifying Grades into one
        MergeTwoValues merge = new MergeTwoValues();
        MergeTwoValues merge1 = new MergeTwoValues();
        MergeTwoValues merge2 = new MergeTwoValues();

        merge.setAttributeIndex("8");
        merge.setFirstValueIndex("1");
        merge.setSecondValueIndex("2");

        merge1.setAttributeIndex("8");
        merge1.setFirstValueIndex("1");
        merge1.setSecondValueIndex("3");

        merge2.setAttributeIndex("8");
        merge2.setFirstValueIndex("2");
        merge2.setSecondValueIndex("3");

        Instances data_new = null;

        try {
            merge.setInputFormat(data_old);
            data_new = Filter.useFilter(data_old, merge);
            merge1.setInputFormat(data_new);
            data_new = Filter.useFilter(data_new, merge1);
            merge2.setInputFormat(data_new);
            data_new = Filter.useFilter(data_new, merge2);
        } catch (Exception ex) {
            Logger.getLogger(KnnTest.class.getName()).log(Level.SEVERE, null, ex);
        }

        return data_new; // return the merge data
    }

    // new created attribute to be the class attribute by placing it in the last position
    public static Instances sortLabels(Instances data_old){

        SortLabels sort = new SortLabels();
        Reorder order = new Reorder();
        Instances data_new = null;
        try {
            order.setAttributeIndices("first-7,9-last,8");
            sort.setInputFormat(data_old);
            data_new = Filter.useFilter(data_old, sort);
            order.setInputFormat(data_new);
            data_new = Filter.useFilter(data_new, order);
        } catch (Exception ex) {
            Logger.getLogger(KnnTest.class.getName()).log(Level.SEVERE, null, ex);
        }

        return data_new; // returns the sorted data
    }
    
    // function to select attributes for new training-testing dataset
    public static Instances attributeSelection(Instances data_old){
        
        Instances data_new =null;
        AttributeSelection selector = new AttributeSelection();
        CfsSubsetEval cfs =new CfsSubsetEval();
        selector.setEvaluator(cfs);
        BestFirst bfs = new BestFirst();
        data_old.setClassIndex(data_old.numAttributes() - 1);
        
        //bfs.setStartSet("11-27");
        selector.setSearch(bfs);
        
        try {
            bfs.setStartSet("1,7,18,27,29");
            selector.setInputFormat(data_old);
            data_new = Filter.useFilter(data_old, selector);

        } catch (Exception ex) {
            Logger.getLogger(KnnTest.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        return data_new;
    }

    // class probability
    public static Instances probClass(Instances data_old){
        ClassConditionalProbabilities prob = new ClassConditionalProbabilities();
        Instances data_new=null;
        
        data_old.setClassIndex(data_old.numAttributes() - 1);
        
        try {
            //prob.setExcludeNumericAttributes(true);
            //prob.setExcludeNominalAttributes(true);
            prob.setInputFormat(data_old);
            data_new = Filter.useFilter(data_old, prob);
        } catch (Exception ex) {
            Logger.getLogger(KnnTest.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        return data_new;
    }

    // Start of program execution
    public static void main(String args[]) throws Exception{

        // Load dataset from MassHousingTrainData.csv File
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("MassHousingTrainData.csv")); // Loading the input file from current directory
        Instances dataset = loader.getDataSet();

        // Discretize the loaded data
        Instances data = discretizedata(dataset);

        // Conversion of data from 'String' to 'Nominal' data
        StringToNominal nom = new StringToNominal();

        nom.setAttributeRange("first-last");
        nom.setInputFormat(data);
        data = Filter.useFilter(data, nom);

        // function call to remove attributes with the data as an argument
        data = removeAtt(data);

        // function call to merge values with the data as an argument
        data = mergeValues(data);

        // function call to sort attributes with the data as an argument
        data = sortLabels(data);

        // function call for attribute selection with the data as an argument
        data = attributeSelection(data);

        //data = probClass(data);
        //data = knnAlgo(data);
        // save ARFF

        // randomize the data to improve the efficency
        Random random = new Random();
        random.setSeed(Integer.MAX_VALUE);
        data.randomize(random);

        // Saving the data to a New File
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File("MassHousingTrainData2.arff"));
        saver.setDestination(new File("MassHousingTrainData2.arff"));
        saver.writeBatch();


        //Instances data = new Instances(datafile);

        // Loading and Reading the new training-testing set
        BufferedReader datafile = readDataFile("MassHousingTrainData2.arff");

        // Creating the new instance object of the training-testing set
        Instances data1 = new Instances(datafile);

        data1.setClassIndex(data1.numAttributes() - 1);

        // Choose a type of validation split
        Instances[][] split = crossValidationSplit(data1, 5);

        // Separate split into training and testing arrays
        Instances[] trainingSplits = split[0];
        Instances[] testingSplits  = split[1];
        //data1.setClassIndex(data1.numAttributes() - 1);

        // Classifier object creation using IBk. Used for classification
        Classifier model = new IBk();
            // Collect every group of predictions for current model in a FastVector
            //FastVector predictions = new FastVector();
            FastVector predictions = new FastVector();
            // For each training-testing split pair, train and test the classifier
            for(int i = 0; i < trainingSplits.length; i++) {
                Evaluation validation = simpleClassify(model, trainingSplits[i], testingSplits[i]);
                predictions.appendElements(validation.predictions());
                //System.out.println(model.toString());
            }

            // Checking and display accuracy of the training-testing results
            double accuracy = calculateAccuracy(predictions, data1);
            System.out.println("Accuracy: " + accuracy + "%");
            //System.out.println(model.getCapabilities());
    }

}
