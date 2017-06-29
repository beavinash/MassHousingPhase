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
import java.io.File;
import java.util.ArrayList;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
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
 * @author aditshah avinash Ashuthosh
 */

public class Knn_pa1a {
    //Discretize the data
    public static Instances discretizedata(Instances data_old) throws Exception{
        
        Discretize discretize = new Discretize();
        discretize.setAttributeIndices("12-last");
        discretize.setInputFormat(data_old);
        Instances data_new = Filter.useFilter(data_old, discretize);
        
        return data_new;
    }

    // function to remove attributes from dataset to improve the speed of preprocessing.
    public static Instances removeAtt(Instances data_old) throws Exception{

        // remove attributes in a dataset to speed up the pre-processing.
        Remove remove =new Remove();
        remove.setAttributeIndices("2-6,11-18,20-32,34-last");//1 7 8 9 10 19  33
        remove.setInputFormat(data_old);
        Instances data_new = Filter.useFilter(data_old, remove);
        
        return data_new; 
    }

    // function to represent A,B,C as 0 and D,F as 1
    public static Instances mergeValues(Instances data_old) throws Exception{

        // creating Instances for merging and classifying Grades into one
        MergeTwoValues merge = new MergeTwoValues();
        MergeTwoValues merge1 = new MergeTwoValues();
        MergeTwoValues merge2 = new MergeTwoValues();
        // Merge A,B
        merge.setAttributeIndex("3");
        merge.setFirstValueIndex("1");
        merge.setSecondValueIndex("2");
        // Merge A,C
        merge1.setAttributeIndex("3");
        merge1.setFirstValueIndex("1");
        merge1.setSecondValueIndex("3");
        //Merge D,F
        merge2.setAttributeIndex("3");
        merge2.setFirstValueIndex("2");
        merge2.setSecondValueIndex("3");

        merge.setInputFormat(data_old);
        Instances data_new = Filter.useFilter(data_old, merge);
        merge1.setInputFormat(data_new);
        data_new = Filter.useFilter(data_new, merge1);
        merge2.setInputFormat(data_new);
        data_new = Filter.useFilter(data_new, merge2);
        
        return data_new; // return the merge data
    }

    // new created attribute in the last position 
    public static Instances sortLabels(Instances data_old) throws Exception{

        SortLabels sort = new SortLabels();
        Reorder order = new Reorder();
        order.setAttributeIndices("first-2,4-last,3");
        sort.setInputFormat(data_old);
        Instances data_new = Filter.useFilter(data_old, sort);
        order.setInputFormat(data_new);
        data_new = Filter.useFilter(data_new, order);

        return data_new; // returns the sorted data placed the class attribute at last position
    }

    // Start of program execution
    public static void main(String args[]) throws Exception{
    // Load dataset from MassHousingTrainData.csv File
        String src="MassHousingTrainData.csv";
        
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(src)); // Loading the data from current directory
        Instances dataset = loader.getDataSet();

        // Discretize the loaded Instances data
        Instances data = discretizedata(dataset);

        // Conversion data from 'String' to 'Nominal' data
        StringToNominal nom = new StringToNominal();
        nom.setAttributeRange("first-last");
        nom.setInputFormat(data);
        data = Filter.useFilter(data, nom);

        // function call to remove attributes with the Instances data as an argument
        data = removeAtt(data);

        // function call to merge values with the data as an argument
        data = mergeValues(data);

        // function call to sort labels with the data as an argument
        data = sortLabels(data);

        // randomize the data to check and improve the efficency gave a best result for max value
        Random random = new Random(Integer.MAX_VALUE);
        data.randomize(random);
        data.setClassIndex(data.numAttributes() - 1);
        // run ibk classifier on the preprocessed data
        Classifier ibk = new IBk();
        data.setClassIndex(data.numAttributes() - 1);
        Evaluation validation =new Evaluation(data);
        //Create train data and test data using 5-fold cross validation to build classifier and test it with test data.
        for(int i = 0; i < 5; i++) {
            Instances traindata = data.trainCV(5, i);
            Instances testdata = data.testCV(5, i);
            ibk.buildClassifier(traindata);
            validation.evaluateModel(ibk, testdata);
        }
        
        ArrayList <Prediction> pred = validation.predictions(); // stores all the prediction of the data
        int count = 0;
        double num_key[] = data.attributeToDoubleArray(data.numAttributes()-1);
        double num_key1[] = data.attributeToDoubleArray(0);
        System.out.println("False Predictions:");
        System.out.println("0:Good(A,B,C) and 1:Bad(D,F)");
        System.out.println("stmt_date \t rm_key \t expected \t predicted");
        // Check which all prediction were false and printing those instances stmt_data, rm_key, expected and predicted
        for(int k=0; k<data.numInstances(); k++){
            if(pred.get(k).predicted()!= pred.get(k).actual() && pred.get(k).actual() == num_key[k]){
                System.out.println(data.get(k).toString(1) + "\t  " + (int)num_key1[k] + "\t\t  " + (int)num_key[k] + "   \t\t   " + (int)pred.get(k).predicted());
                count ++;
            }
        }
        //printing total incorrect classification for justification of printed instances
        System.out.println("total incorrect: " + count);
        //prints the report of the classification
        System.out.println(validation.toSummaryString(false));
        //prints the confusion matrix.
        System.out.println(validation.toMatrixString());
        if(args.length > 0){
            System.out.println("Now Performing Testing of the input test dataset: ");
            String src_new = args[0];
            CSVLoader loader_new = new CSVLoader();
            loader_new.setSource(new File(src_new)); // Loading the data from current directory
            Instances data_test = loader.getDataSet();

            data_test = discretizedata(data_test);

            StringToNominal nom_new = new StringToNominal();
            nom_new.setAttributeRange("first-last");
            nom_new.setInputFormat(data_test);
            data_test = Filter.useFilter(data_test, nom_new);

            data_test = removeAtt(data_test);

            data_test = mergeValues(data_test);

            data_test = sortLabels(data_test);
            
            data_test.setClassIndex(data_test.numAttributes() - 1);
            Classifier ibk_test = new IBk();
            Evaluation validation_new = new Evaluation(data);
            ibk_test.buildClassifier(data);
            validation_new.evaluateModel(ibk_test, data_test);
            System.out.println(validation_new.toSummaryString(false));
            System.out.println(validation_new.toMatrixString());
        }
    }

}



