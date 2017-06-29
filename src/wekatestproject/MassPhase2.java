/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekatestproject;



import java.io.File;
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
import weka.filters.unsupervised.attribute.Copy;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.Reorder;
import weka.filters.unsupervised.attribute.SortLabels;
import weka.filters.unsupervised.attribute.StringToNominal;
import weka.filters.unsupervised.instance.RemoveWithValues;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author aditshah
 */
public class MassPhase2 {
    
    //Convert to nominal
    private Instances dataToNominal(Instances data) throws Exception{
        
        NumericToNominal num = new NumericToNominal();
        num.setAttributeIndices("1");
        num.setInputFormat(data);
        data = Filter.useFilter(data, num);
        return data;
    }
    
    //Remove Data less than 10 years
    private Instances removeData(Instances data, String range, String indices) throws Exception{
        RemoveWithValues rm = new RemoveWithValues();
        //rm.setAttributeIndex(range);
        rm.setAttributeIndex(indices);
        rm.setNominalIndices(range);
        rm.setInputFormat(data);
        data = Filter.useFilter(data, rm);
        return data;
    }
    
    //String to nominal
    private Instances string2nominal(Instances data)throws Exception{
        StringToNominal sn = new StringToNominal();
        sn.setAttributeRange("7");
        sn.setInputFormat(data);
        data = Filter.useFilter(data, sn);
        return data;
    }
    
    private Instances sortLabels(Instances data_old) throws Exception{

        SortLabels sort = new SortLabels();
        Reorder order = new Reorder();
        order.setAttributeIndices("first-2,4-last,3");
        sort.setInputFormat(data_old);
        Instances data_new = Filter.useFilter(data_old, sort);
        order.setInputFormat(data_new);
        data_new = Filter.useFilter(data_new, order);

        return data_new; // returns the sorted data placed the class attribute at last position
    }
    
    private Instances removeAtt(Instances data_old) throws Exception{

        // remove attributes in a dataset to speed up the pre-processing.
        Remove remove =new Remove();
        //remove.setAttributeIndices("2-6,9-10,15,18,20-52");//1 7 8 9 10 19  33
        remove.setAttributeIndices("2-6,18,20-52");
        remove.setInputFormat(data_old);
        Instances data_new = Filter.useFilter(data_old, remove);
        
        return data_new; 
    }
    
    private Instances copyAtt(Instances data_old) throws Exception{

        // remove attributes in a dataset to speed up the pre-processing.
        Copy copy = new Copy();
        copy.setAttributeIndices("1");//1 7 8 9 10 19  33
        copy.setInputFormat(data_old);
        Instances data_new = Filter.useFilter(data_old, copy);
        
        return data_new; 
    }
    
    //5-fold cross validation
    private double[] validate(Instances data) throws Exception{
        double percent;
        Random random = new Random(Integer.MAX_VALUE);
        data.randomize(random);
        Remove remove =new Remove();
        //remove.setAttributeIndices("2-6,9-10,15,18,20-52");//1 7 8 9 10 19  33
        remove.setAttributeIndices("1-2,13");
        remove.setInputFormat(data);
        Instances data_new = Filter.useFilter(data, remove);
        data_new.setClassIndex(data_new.numAttributes() - 1);
        // run ibk classifier on the preprocessed data
        Classifier ibk = new IBk(4);
        Evaluation validation =new Evaluation(data_new);
        //Create train data and test data using 5-fold cross validation to build classifier and test it with test data.
        for(int i = 0; i < 5; i++) {
            Instances traindata = data_new.trainCV(5, i);
            Instances testdata = data_new.testCV(5, i);
            ibk.buildClassifier(traindata);
            validation.evaluateModel(ibk, testdata);
        }
        ArrayList <Prediction> pred = validation.predictions(); // stores all the prediction of the data
        double count = 0;
        double count_tot[] = new double[10];
        double num_key[] = data.attributeToDoubleArray(data.numAttributes()-1);
        double num_key1[] = data.attributeToDoubleArray(12);
        System.out.println("False Predictions:");
        System.out.println("0:A, 1:B, 2:C, 3:D, 4:F");
        System.out.println("stmt_date \t rm_key \t expected \t predicted");
        // Check which all prediction were false and printing those instances stmt_data, rm_key, expected and predicted
        for(int k=0; k<data.numInstances(); k++){
            if(num_key[k] == 0.0){
                count_tot[0] ++;
            }
            if(num_key[k] == 1.0){
                count_tot[1] ++;
            }
            if(num_key[k] == 2.0){
                count_tot[2] ++;
            }
            if(num_key[k] == 3.0){
                count_tot[3] ++;
            }
            if(num_key[k] == 4.0){
                count_tot[4] ++;
            }
            if(pred.get(k).predicted()!= pred.get(k).actual() && pred.get(k).actual() == num_key[k] && num_key[k] == 0.0){
                System.out.println(data.get(k).toString(1) + "\t  " + (int)num_key1[k] + "\t\t  " + (int)num_key[k] + "   \t\t   " + (int)pred.get(k).predicted());
                count_tot[5] ++;
            }
            if(pred.get(k).predicted()!= pred.get(k).actual() && pred.get(k).actual() == num_key[k] && num_key[k] == 1.0){
                System.out.println(data.get(k).toString(1) + "\t  " + (int)num_key1[k] + "\t\t  " + (int)num_key[k] + "   \t\t   " + (int)pred.get(k).predicted());
                count_tot[6] ++;
            }
            if(pred.get(k).predicted()!= pred.get(k).actual() && pred.get(k).actual() == num_key[k] && num_key[k] == 2.0){
                System.out.println(data.get(k).toString(1) + "\t  " + (int)num_key1[k] + "\t\t  " + (int)num_key[k] + "   \t\t   " + (int)pred.get(k).predicted());
                count_tot[7] ++;
            }
            if(pred.get(k).predicted()!= pred.get(k).actual() && pred.get(k).actual() == num_key[k] && num_key[k] == 3.0){
                System.out.println(data.get(k).toString(1) + "\t  " + (int)num_key1[k] + "\t\t  " + (int)num_key[k] + "   \t\t   " + (int)pred.get(k).predicted());
                count_tot[8] ++;
            }
            if(pred.get(k).predicted()!= pred.get(k).actual() && pred.get(k).actual() == num_key[k] && num_key[k] == 4.0){
                System.out.println(data.get(k).toString(1) + "\t  " + (int)num_key1[k] + "\t\t  " + (int)num_key[k] + "   \t\t   " + (int)pred.get(k).predicted());
                count_tot[9] ++;
            }
        }
        //prints the report of the classification
        System.out.println(validation.toSummaryString(false));
        return count_tot;
    }
    
    public static void main(String args[]) throws Exception{
        String src="MassHousingTrainData.csv";
        MassPhase2 m = new MassPhase2();
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(src)); // Loading the data from current directory
        Instances dataset = loader.getDataSet();
        Instances data = m.copyAtt(dataset);
        data = m.dataToNominal(data);
        data = m.string2nominal(data);
        String range="1-64,333-344,348,350,355,358,361-368";
        data = m.removeData(data, range, "1");
        data = m.string2nominal(data);
        data = m.removeAtt(data);
        data = m.sortLabels(data);
        range = "1,3,10-13,15-20,25-28,30-36,38-last";
        Instances datatest1 = m.removeData(data, range, "2");
        datatest1 = m.removeData(datatest1, "32,34,36,39,45,46,47,52-139,162,212,259,268,360", "1");
        range = "1-3,5,11-14,16-21,26-29,31-37,39-last";
        Instances datatest2 = m.removeData(data, range, "2");
        datatest2 = m.removeData(datatest2, "34,36,39-139,144,162,212,245,249,259,268,360", "1");
        range = "2-3,5-6,12-15,17-21,27-30,32-38,40-last";//"7,22,8,23,4,9,24,10,25,1,11,26,16,31,39";
        Instances datatest3 = m.removeData(data, range, "2");
        datatest3 = m.removeData(datatest3, "39-51,53-139,144,212,245,249,259,268", "1");
        range = "2,5-7,13-16,18-22,28-31,33-39,41-last";//"8,23,4,9,24,10,25,1,11,26,12,27,3,17,32,40";
        Instances datatest4 = m.removeData(data, range, "2");
        datatest4 = m.removeData(datatest4, "40-51,53-139,141,142,144,148,212,245,249,259,268", "1");
        range = "2-3,5-8,14-17,19-23,29-32,34,35,37-40,42-last";//"4,9,24,10,25,1,11,26,12,27,13,28,36,18,33,41";
        Instances datatest5 = m.removeData(data, range, "2");
        datatest5 = m.removeData(datatest5, "40-51,55,57-58,61,62,65-139,141,142,144,148,212,245,249,259,307", "1");
        range = "3-9,15-18,21-24,30-33,38-41"; // 10,25,1,11,26,12,27,13,28,36,2,14,29,37,19,20,34,35,42
        Instances datatest6 = m.removeData(data, range, "2"); 
        datatest6 = m.removeData(datatest6, "40-51,55,57,62,65,68-139,141,142,144,148,245,249,259,344", "1"); 
        ArffSaver saver = new ArffSaver();
        saver.setInstances(datatest3);
        saver.setFile(new File("MassHousingPhase2.arff"));
        saver.setDestination(new File("MassHousingPhase2.arff"));
        saver.writeBatch();
        double percentage1[]=new double[10];
        double percentage2[]=new double[10];
        double percentage3[]=new double[10];
        double percentage4[]=new double[10];
        double percentage5[]=new double[10];
        double percentage6[]=new double[10];
        percentage1= m.validate(datatest1);
        System.out.println("1st pass A: " + ((percentage1[0]-percentage1[5])/percentage1[0]) *100 + "%");
        System.out.println("1st pass B: " + ((percentage1[1]-percentage1[6])/percentage1[1]) *100 + "%");
        System.out.println("1st pass C: " + ((percentage1[2]-percentage1[7])/percentage1[2]) *100 + "%");
        System.out.println("1st pass D: " + ((percentage1[3]-percentage1[8])/percentage1[3]) *100 + "%");
        System.out.println("1st pass F: " + ((percentage1[4]-percentage1[9])/percentage1[4]) *100 + "%");
        percentage2= m.validate(datatest2);
        System.out.println("2nd pass A: " + ((percentage2[0]-percentage2[5])/percentage2[0]) *100 + "%");
        System.out.println("2nd pass B: " + ((percentage2[1]-percentage2[6])/percentage2[1]) *100 + "%");
        System.out.println("2nd pass C: " + ((percentage2[2]-percentage2[7])/percentage2[2]) *100 + "%");
        System.out.println("2nd pass D: " + ((percentage2[3]-percentage2[8])/percentage2[3]) *100 + "%");
        System.out.println("2nd pass F: " + ((percentage2[4]-percentage2[9])/percentage2[4]) *100 + "%");
        percentage3= m.validate(datatest3);
        System.out.println("3rd pass A: " + ((percentage3[0]-percentage3[5])/percentage3[0]) *100 + "%");
        System.out.println("3rd pass B: " + ((percentage3[1]-percentage3[6])/percentage3[1]) *100 + "%");
        System.out.println("3rd pass C: " + ((percentage3[2]-percentage3[7])/percentage3[2]) *100 + "%");
        System.out.println("3rd pass D: " + ((percentage3[3]-percentage3[8])/percentage3[3]) *100 + "%");
        System.out.println("3rd pass F: " + ((percentage3[4]-percentage3[9])/percentage3[4]) *100 + "%");
        percentage4= m.validate(datatest4);
        System.out.println("4th pass A: " + ((percentage4[0]-percentage4[5])/percentage4[0]) *100 + "%");
        System.out.println("4th pass B: " + ((percentage4[1]-percentage4[6])/percentage4[1]) *100 + "%");
        System.out.println("4th pass C: " + ((percentage4[2]-percentage4[7])/percentage4[2]) *100 + "%");
        System.out.println("4th pass D: " + ((percentage4[3]-percentage4[8])/percentage4[3]) *100 + "%");
        System.out.println("4th pass F: " + ((percentage4[4]-percentage4[9])/percentage4[4]) *100 + "%");
        percentage5= m.validate(datatest5);
        System.out.println("5th pass A: " + ((percentage5[0]-percentage5[5])/percentage5[0]) *100 + "%");
        System.out.println("5th pass B: " + ((percentage5[1]-percentage5[6])/percentage5[1]) *100 + "%");
        System.out.println("5th pass C: " + ((percentage5[2]-percentage5[7])/percentage5[2]) *100 + "%");
        System.out.println("5th pass D: " + ((percentage5[3]-percentage5[8])/percentage5[3]) *100 + "%");
        System.out.println("5th pass F: " + ((percentage5[4]-percentage5[9])/percentage5[4]) *100 + "%");
        percentage6= m.validate(datatest6);
        System.out.println("6th pass A: " + ((percentage6[0]-percentage6[5])/percentage6[0]) *100 + "%");
        System.out.println("6th pass B: " + ((percentage6[1]-percentage6[6])/percentage6[1]) *100 + "%");
        System.out.println("6th pass C: " + ((percentage6[2]-percentage6[7])/percentage6[2]) *100 + "%");
        System.out.println("6th pass D: " + ((percentage6[3]-percentage6[8])/percentage6[3]) *100 + "%");
        System.out.println("6th pass F: " + ((percentage6[4]-percentage6[9])/percentage6[4]) *100 + "%");
        double total_a =percentage1[0] + percentage2[0] + percentage3[0] + percentage4[0] + percentage5[0] + percentage6[0];
        double total_right = (percentage1[0]-percentage1[5]) + (percentage2[0]-percentage2[5]) + (percentage3[0]-percentage3[5]) + (percentage4[0]-percentage4[5]) + (percentage5[0]-percentage5[5]) + (percentage6[0]-percentage6[5]);
        System.out.println("Average A accuracy: " + total_right/total_a * 100 + "%");
        double total_b =percentage1[1] + percentage2[1] + percentage3[1] + percentage4[1] + percentage5[1] + percentage6[1];
        total_right = (percentage1[1]-percentage1[6]) + (percentage2[1]-percentage2[6]) + (percentage3[1]-percentage3[6]) + (percentage4[1]-percentage4[6]) + (percentage5[1]-percentage5[6]) + (percentage6[1]-percentage6[6]);
        System.out.println("Average B accuracy: " + total_right/total_b * 100 + "%");
        double total_c =percentage1[2] + percentage2[2] + percentage3[2] + percentage4[2] + percentage5[2] + percentage6[2];
        total_right = (percentage1[2]-percentage1[7]) + (percentage2[2]-percentage2[7]) + (percentage3[2]-percentage3[7]) + (percentage4[2]-percentage4[7]) + (percentage5[2]-percentage5[7]) + (percentage6[2]-percentage6[7]);
        System.out.println("Average C accuracy:" + total_right/total_c * 100 + "%");
        double total_d =percentage1[3] + percentage2[3] + percentage3[3] + percentage4[3] + percentage5[3] + percentage6[3];
        total_right = (percentage1[3]-percentage1[8]) + (percentage2[3]-percentage2[8]) + (percentage3[3]-percentage3[8]) + (percentage4[3]-percentage4[8]) + (percentage5[3]-percentage5[8]) + (percentage6[3]-percentage6[8]);
        System.out.println("Average D accuracy:" + total_right/total_d * 100 + "%");
        double total_f =percentage1[4] + percentage2[4] + percentage3[4] + percentage4[4] + percentage5[4] + percentage6[4];
        total_right = (percentage1[4]-percentage1[9]) + (percentage2[4]-percentage2[9]) + (percentage3[4]-percentage3[9]) + (percentage4[4]-percentage4[9]) + (percentage5[4]-percentage5[9]) + (percentage6[4]-percentage6[9]);
        System.out.println("Average F accuracy:" + total_right/total_f * 100 + "%");
    }
}

