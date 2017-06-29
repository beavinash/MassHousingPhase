
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.supervised.attribute.ClassConditionalProbabilities;
import weka.filters.unsupervised.attribute.Copy;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.Reorder;
import weka.filters.unsupervised.attribute.SortLabels;
import weka.filters.unsupervised.attribute.StringToNominal;



/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
/**
 *
 * @author tirathshah
 */
public class MassHousingPhase3 {

    private Instances loadingData(String src) throws Exception {
        //MassHousingPhase3 m = new MassHousingPhase3();
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(src)); // Loading the data from current directory
        Instances dataset = loader.getDataSet();
        

        return dataset;
    }

    private void savingData(Instances newTrainData, String src) throws IOException {
        ArffSaver saver = new ArffSaver();
        saver.setInstances(newTrainData);
        saver.setFile(new File(src));
        saver.setDestination(new File(src));
        saver.writeBatch();

    }
    
    private Instances sortLabels(Instances data_old) throws Exception{

        SortLabels sort = new SortLabels();
        Reorder order = new Reorder();
        order.setAttributeIndices("2-last,1");
        sort.setInputFormat(data_old);
        Instances data_new = Filter.useFilter(data_old, sort);
        order.setInputFormat(data_new);
        data_new = Filter.useFilter(data_new, order);

        return data_new; // returns the sorted data placed the class attribute at last position
    }
    
    private Instances removeAtt(Instances data_old) throws Exception{

        // remove attributes in a dataset to speed up the pre-processing.
        Remove remove =new Remove();
        remove.setAttributeIndices("1-7,9,10,15,18,20-52");
        remove.setInputFormat(data_old);
        Instances data_new = Filter.useFilter(data_old, remove);
        
        return data_new; 
    }
    
    private Instances randomInst(Instances data)throws Exception{
        Random random = new Random(Integer.MAX_VALUE);
        data.randomize(random);
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }
    
    private Instances normalizeData(Instances data)throws Exception{
        Normalize norm = new Normalize();
        norm.setScale(1.0);
        norm.setTranslation(0.0);
        norm.setInputFormat(data);
        Instances data_new = Filter.useFilter(data, norm);
        return data_new;
    }
    
    private Instances string2nominal(Instances data)throws Exception{
        StringToNominal sn = new StringToNominal();
        sn.setAttributeRange("first-last");
        sn.setInputFormat(data);
        data = Filter.useFilter(data, sn);
        return data;
    }
    
    private void printResult(Evaluation validation, Instances data) throws Exception{
        ArrayList <Prediction> pred = validation.predictions(); // stores all the prediction of the data
        int count = 0;
        double num_key[] = data.attributeToDoubleArray(7);
        double num_key1[] = data.attributeToDoubleArray(0);
        System.out.println("False Predictions:");
        System.out.println("0:Good(A,B,C) and 1:Bad(D,F)");
        System.out.println("stmt_date \t rm_key \t expected \t predicted");
        // Check which all prediction were false and printing those instances stmt_data, rm_key, expected and predicted
        for(int k=0; k<data.numInstances(); k++){
            if(pred.get(k).predicted()!= pred.get(k).actual()){
                System.out.println(data.get(k).toString(6) + "\t  " + (int)num_key1[k] + "\t\t  " + (int)pred.get(k).actual() + "   \t\t   " + (int)pred.get(k).predicted());
                count++;
            }
        }
        //printing total incorrect classification for justification of printed instances
        System.out.println("total incorrect: " + count);
        //prints the report of the classification
        System.out.println(validation.toSummaryString(false));
        //prints the confusion matrix.
        System.out.println(validation.toMatrixString());
    }
    
    private Instances copyAtt(Instances data_old) throws Exception{

        Copy copy = new Copy();
        copy.setAttributeIndices("1");
        copy.setInputFormat(data_old);
        Instances data_new = Filter.useFilter(data_old, copy);
        
        return data_new; 
    }
    
    private Instances classConditional(Instances data)throws Exception{
        //StringToNominal sn = new StringToNominal();
        ClassConditionalProbabilities c = new ClassConditionalProbabilities();
        //c.setAttributeRange("first-last");
        data.setClassIndex(data.numAttributes()-1);
        c.setNominalConversionThreshold(-1);
        c.setInputFormat(data);
        data = Filter.useFilter(data, c);
        return data;
    }
    
    // Main Function. Program starts here
    public static void main(String[] args) throws IOException, Exception {
        MassHousingPhase3 m = new MassHousingPhase3();
        // massnewtraindata
        String src = "blankless_masstraindata(3351).csv";
        Instances newTrainData = m.loadingData(src);
        Instances data1 = newTrainData;
        newTrainData = m.removeAtt(newTrainData);
        newTrainData = m.sortLabels(newTrainData);
        newTrainData = m.string2nominal(newTrainData);
        newTrainData = m.normalizeData(newTrainData);
        //newTrainData = m.randomInst(newTrainData);
        //newTrainData = m.classConditional(newTrainData);
        
        // Test_Set_New_rm_key
        String src2 = "blankless_testdata(1035).csv";
        Instances testNewData = m.loadingData(src2);
        Instances data2 = testNewData;
        testNewData = m.removeAtt(testNewData);
        testNewData = m.sortLabels(testNewData);
        testNewData = m.string2nominal(testNewData);
        testNewData = m.normalizeData(testNewData);
        //testNewData = m.randomInst(testNewData);
        //testNewData = m.classConditional(testNewData);
        
        // testdata(885)
        String src3 = "Test_Set_New_rm_key.csv";
        Instances testData = m.loadingData(src3);
        Instances data3 = testData;
        testData = m.removeAtt(testData);
        testData = m.sortLabels(testData);
        testData = m.string2nominal(testData);
        testData = m.normalizeData(testData);
        //testData = m.randomInst(testData);
        //testData = m.classConditional(testData);
        //testData = m.string2nominal(testData);
        // updatedtraining(1575)
        //String src4 = "updatedtraining(1575).csv";
        //Instances updatedData = m.loadingData(src4);
        
        String src4 = "MassHousingTrainData.csv";
        Instances train = m.loadingData(src4);
        Instances data4 = train;
        train = m.removeAtt(train);
        train = m.sortLabels(train);
        train = m.string2nominal(train);
        train = m.normalizeData(train);
        //train = m.randomInst(train);
        //train = m.classConditional(train);
        
        m.savingData(newTrainData, "blankless_masstraindata(3351).arff");
        m.savingData(testNewData, "blankless_testdata(1035).arff");
        m.savingData(testData,  "Test_Set_New_rm_key.arff");
        m.savingData(train,  "MassHousingTrainData.arff");
        
        //m.savingData(updatedData, "updatedtraining(1575).arff");
        testNewData.setClassIndex(testNewData.numAttributes()-1);
        newTrainData.setClassIndex(newTrainData.numAttributes()-1);
        Classifier ibk_test = new IBk(7);
        Evaluation validation_new = new Evaluation(newTrainData);
        ibk_test.buildClassifier(newTrainData);
        validation_new.evaluateModel(ibk_test, testNewData);
        //System.out.println(validation_new.toSummaryString(false));
        //System.out.println(validation_new.toMatrixString());
        m.printResult(validation_new, data2);
        
        testData.setClassIndex(testData.numAttributes()-1);
        train.setClassIndex(train.numAttributes()-1);
        Classifier ibk_final_test = new IBk(7);
        Evaluation validation = new Evaluation(train);
        ibk_final_test.buildClassifier(train);
        validation.evaluateModel(ibk_final_test, testData);
        //System.out.println(validation.toSummaryString(false));
        //System.out.println(validation.toMatrixString());
        m.printResult(validation, data3);

    }
}
 