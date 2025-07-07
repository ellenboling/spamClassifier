import java.io.*;
import java.util.*;

//This class creates a binary classification tree that allows
//users to train/load a model and classify data. 
public class Classifier {  
    private ClassifierNode root;
    
    private static class ClassifierNode{
        public String feature;
        public double threshold;
        public ClassifierNode left;
        public ClassifierNode right;
        public String label;
        public TextBlock data;


        public ClassifierNode(String label, TextBlock input){
            this.label = label;
            this.data = input;
        }

        public ClassifierNode(String label){
            this.label = label;
        }

        public ClassifierNode(String feature, double threshold, ClassifierNode left, ClassifierNode right){
            this.feature = feature;
            this.threshold = threshold;
            this.left = left;
            this.right = right;
        }
        
        public boolean checkIfLeafNode(){
            return label != null;
        }
    }   

    //Behavior: Loads classifier from file connected to Scanner.
    //Exceptions: throws IllegalArgumentException if input is null.
    //Exceptions: throws IllegalStateException if root is null.
    //Parameters: input to access user input.
    public Classifier(Scanner input) {
        if(input == null){
            throw new IllegalArgumentException();
        }
        root = buildTree(input);
        if(root == null){
            throw new IllegalStateException();
        }
    }

    private ClassifierNode buildTree(Scanner input){
        if(!input.hasNextLine()){
            return null;
        }
        String line = input.nextLine();
        if(line.startsWith("Feature: ")){
            String feature = line.substring("Feature: ".length());
            if(!input.hasNextLine()){
                throw new IllegalStateException();
            }
            double threshold = Double.parseDouble(input.nextLine().substring("Threshold: ".length()));
            ClassifierNode left = buildTree(input);
            ClassifierNode right = buildTree(input);
            return new ClassifierNode(feature, threshold, left, right);
        }else{
            return new ClassifierNode(line);
        }
    } 
        
    //Behavior: creates and trains classifier from input data and labels.
    //Exceptions: throws IllegalArgumentException if data or labels is null, 
    //if data length is wrong, or if data or labels is empty.
    //Parameters: data to access training data
    //Parameters: labels to access labels list
    public Classifier(List<TextBlock> data, List<String> labels) {
        if(data == null || labels == null || data.size() != labels.size() || (data.isEmpty() || labels.isEmpty())){
            throw new IllegalArgumentException();
        }
        root = new ClassifierNode(labels.get(0), data.get(0));
        for(int i = 1; i < data.size(); i++){
            root = train(root, data.get(i), labels.get(i));
        }
    }

    private ClassifierNode train(ClassifierNode node, TextBlock input, String label){
        if(node.checkIfLeafNode()){
            if(node.label.equals(label)){
                return node;
            }

            String feature = input.findBiggestDifference(node.data);
            double inputTime = input.get(feature);
            double nodeTime = node.data.get(feature);
            double threshold = midpoint(inputTime, nodeTime);

            ClassifierNode newDecision = new ClassifierNode(feature, threshold, null, null);

            if(inputTime < threshold){
                newDecision.left = new ClassifierNode(label, input);
                newDecision.right = node;
            }else{
                newDecision.left = node;
                newDecision.right = new ClassifierNode(label, input);
            }
            return newDecision;
        }
        double value = input.get(node.feature);
        if(value < node.threshold){
            node.left = train(node.left, input, label);
        }else{
            node.right = train(node.right, input, label);
        }
        return node;
    }


    //Behavior: evaluates input data to determine if less than threshold.
    //Exceptions: throws IllegalArgumentException if input is null.
    //Parameters: input to access user input
    //Returns: returns appropriate label that classifier predicts 
    public String classify(TextBlock input) {
        if(input == null){
            throw new IllegalArgumentException();
        }
        return classify(root, input);
    }

    private String classify(ClassifierNode node, TextBlock input){
        if(node.checkIfLeafNode()){
            return node.label;
        }
        double continued = input.get(node.feature);
        if(continued < node.threshold){
            return classify(node.left, input);
        }else{
            return classify(node.right, input);
        }
    }

    //Behavior: saves current classifier to printstream.
    //Exceptions: throws IllegalArgumentException if output is null.
    //Parameters: output to access where classifier is saved.
    public void save(PrintStream output) {
        if(output == null){
            throw new IllegalArgumentException();
        }
        save(output, root);
     }

     private void save(PrintStream output, ClassifierNode node){
        if(node.checkIfLeafNode()){
            output.println(node.label);
        }else{
            output.println("Feature: " + node.feature);
            output.println("Threshold: " + node.threshold);
            save(output, node.left);
            save(output, node.right);
        }
     }



    ////////////////////////////////////////////////////////////////////
    // PROVIDED METHODS - **DO NOT MODIFY ANYTHING BELOW THIS LINE!** //
    ////////////////////////////////////////////////////////////////////

    // Helper method to calcualte the midpoint of two provided doubles.
    private static double midpoint(double one, double two) {
        return Math.min(one, two) + (Math.abs(one - two) / 2.0);
    }    

    // Behavior: Calculates the accuracy of this model on provided Lists of 
    //           testing 'data' and corresponding 'labels'. The label for a 
    //           datapoint at an index within 'data' should be found at the 
    //           same index within 'labels'.
    // Exceptions: IllegalArgumentException if the number of datapoints doesn't match the number 
    //             of provided labels
    // Returns: a map storing the classification accuracy for each of the encountered labels when
    //          classifying
    // Parameters: data - the list of TextBlock objects to classify. Should be non-null.
    //             labels - the list of expected labels for each TextBlock object. 
    //             Should be non-null.
    public Map<String, Double> calculateAccuracy(List<TextBlock> data, List<String> labels) {
        // Check to make sure the lists have the same size (each datapoint has an expected label)
        if (data.size() != labels.size()) {
            throw new IllegalArgumentException(
                    String.format("Length of provided data [%d] doesn't match provided labels [%d]",
                                  data.size(), labels.size()));
        }
        
        // Create our total and correct maps for average calculation
        Map<String, Integer> labelToTotal = new HashMap<>();
        Map<String, Double> labelToCorrect = new HashMap<>();
        labelToTotal.put("Overall", 0);
        labelToCorrect.put("Overall", 0.0);
        
        for (int i = 0; i < data.size(); i++) {
            String result = classify(data.get(i));
            String label = labels.get(i);

            // Increment totals depending on resultant label
            labelToTotal.put(label, labelToTotal.getOrDefault(label, 0) + 1);
            labelToTotal.put("Overall", labelToTotal.get("Overall") + 1);
            if (result.equals(label)) {
                labelToCorrect.put(result, labelToCorrect.getOrDefault(result, 0.0) + 1);
                labelToCorrect.put("Overall", labelToCorrect.get("Overall") + 1);
            }
        }

        // Turn totals into accuracy percentage
        for (String label : labelToCorrect.keySet()) {
            labelToCorrect.put(label, labelToCorrect.get(label) / labelToTotal.get(label));
        }
        return labelToCorrect;
    }
}