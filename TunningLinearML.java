package examples;

import evaluation.tuning.searchers.GridSearcher;
import tsml.classifiers.distance_based.utils.collections.params.iteration.GridSearch;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Instance;
import weka.core.Instances;

import java.io.FileReader;
import java.util.Random;

public class Lab7_TunningSV {
    public static void main(String args[]) throws Exception {
        SMO smo = new SMO();
        //RBFKernel
        RBFKernel rbfKernel = new RBFKernel();
        rbfKernel.setGamma(0.1);
        smo.setKernel(rbfKernel);
        smo.setC(10);

        SMO smo2 = new SMO();
        PolyKernel polyKernel = new PolyKernel();
        polyKernel.setExponent(1.0);
        smo2.setKernel(polyKernel);

        MultilayerPerceptron multilayerPerceptron = new MultilayerPerceptron();
        multilayerPerceptron.setHiddenLayers(String.valueOf(1.0));
        multilayerPerceptron.setLearningRate(0.1);
        multilayerPerceptron.setMomentum(0.1);
        multilayerPerceptron.setDecay(false);


        //dataset file path
        String DATASETPATH = "hayes-roth.arff";
        //read the data
        FileReader reader = null;
        reader = new FileReader(DATASETPATH);

        //create data instances
        Instances data = null;
        data = new Instances(reader);

        //first set randomize your data
        data.randomize(new java.util.Random(0));
        //set the last class as the one we are trying to predict
        data.setClassIndex(data.numAttributes() - 1);

        //resampling
        //second step split the data into 70% training set and 30% test set
        int trainSize = (int) Math.round(data.numInstances() * 0.7);
        int testSize = (int) data.numInstances() - trainSize;
        //CREATE THE INSTANCES FOR THE TEST AND TRAIN SET
        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, testSize);

        //build Support Vector Classifier wih RBFKernel
        smo.buildClassifier(train);
        smo2.buildClassifier(train);
        multilayerPerceptron.buildClassifier(train);

        //RBF kernel Evaluation
        Evaluation evalRBF = new Evaluation(train);
        evalRBF.evaluateModel(smo,test);
        System.out.println(evalRBF.toSummaryString("\nResults",false));

        //POly Kernel Evaluation
        Evaluation evalPoly = new Evaluation(train);
        evalPoly.evaluateModel(smo2,test);
        System.out.println(evalPoly.toSummaryString("\nResults",false));

        //Neural network Evaluation
        Evaluation evalNeuralNet = new Evaluation(train);
        evalNeuralNet.evaluateModel(multilayerPerceptron,test);
        System.out.println(evalNeuralNet.toSummaryString("\nResults",false));


    public static double tunningClassifiersRBFKernel(Instances train){
        /* This function finds the best min Error value, the best gamma, and the best C value
        for the support Support Vector Machine algorithm with the RBF Kernel*/
        SMO cls = new SMO();
        ///RBF Kernel tunning
        RBFKernel rbfKernel = new RBFKernel();
        cls.setKernel(rbfKernel);
        double minError =  1000000;
        double bestGamma = 0;
        double bestC = 0;
        double[] gamma = {0.001, 0.01, 0.1, 1, 10};
        double[] C = {0.001, 0.01, 0.1, 1, 10};
        for(int i=0; i<gamma.length;i++){
            for(int j=0; j<C.length; j++){
                rbfKernel.setGamma(gamma[i]);
                cls.setC(C[j]);
                try {
                    cls.buildClassifier(train);
                    Evaluation eval = new Evaluation(train);
                    eval.crossValidateModel(cls, train, 10, new Random());
                    double error = eval.errorRate();
                    System.out.println("Error rate for ZeroR on train CV RBFKernel =" + error);
                    if (error < minError){
                        minError = error;
                        bestGamma = gamma[i];
                        bestC = C[j];
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }

            }
        }
        System.out.println("Best Gamma " + bestGamma+ " Best C " + bestC);
        return minError;
    }
    public static double tunningClassifiersPolyKernel(Instances train){
        /* This function finds the best min error, the best exponent
        for the support Support Vector Machine algorithm using the PolyKernel*/
        }
        SMO cls = new SMO();
        ///Poly Kernel tunning
        PolyKernel polyKernel = new PolyKernel();
        cls.setKernel(polyKernel);
        double minError =  1000000;
        double bestExponent = 0;
        double[] exponents = {0.001, 0.01, 0.1, 1, 5, 10, 15, 20, 25, 35.5, 40, 50, 100, 150, 500, 905.75, 1000};
        for(int i=0; i<exponents.length;i++){
            polyKernel.setExponent(exponents[i]);

            try {
                cls.buildClassifier(train);
                Evaluation eval = new Evaluation(train);
                eval.crossValidateModel(cls, train, 10, new Random());
                double error = eval.errorRate();
                System.out.println("Error rate for ZeroR on train CV PolyKernel =" + error);
                if (error < minError){
                    minError = error;
                    bestExponent = exponents[i];
                    System.out.println("Best Exponent " + bestExponent);
                }
            } catch (Exception e) {
                e.printStackTrace();
            }

        }


        return minError;
    }

    public static double tunningClassifiersMLP(Instances train){
        /* This function finds the best hidden layers, the best learning rate, best momentum, best decay
        for the support Multilayer Perceptron which is a Neural Network algorithm*/

        MultilayerPerceptron cls = new MultilayerPerceptron();

        ///Multilayer Perception tunning
        double minError =  1000000;

        double bestHiddenlayers = 0;
        double bestLearningRate = 0;
        double bestMomentum = 0;
        boolean bestDecay = false;


        double[] hiddenlayers = {0.001, 0.01, 0.1, 1, 10};
        double[] learningRate = {0.001, 0.01, 0.1, 1, 10};
        double[] momentum = {0.001, 0.01, 0.1, 1, 10};
        boolean[] decay ={true, false};


        for(int i=0; i < hiddenlayers.length;i++){
            for(int j=0; j < learningRate.length; j++){
                for(int k=0; k<momentum.length; k++){
                    for(int b=0; b<decay.length; b++){

                            cls.setHiddenLayers(String.valueOf(hiddenlayers[i]));
                            cls.setLearningRate(learningRate[j]);
                            cls.setMomentum(momentum[k]);
                            cls.setDecay(decay[b]);

                            try {

                                cls.buildClassifier(train);

                                Evaluation eval = new Evaluation(train);
                                eval.crossValidateModel(cls, train, 10, new Random());
                                double error = eval.errorRate();
                                System.out.println("Error rate for ZeroR on train CV RBFKernel =" + error);

                                if (error < minError){
                                    minError = error;

                                    bestHiddenlayers = hiddenlayers[i];
                                    bestLearningRate = learningRate[j];
                                    bestMomentum = momentum[k];
                                    bestDecay = decay[b];

                                }

                            } catch (Exception e) {
                                e.printStackTrace();
                            }
                        }
                    }
                }
            
            }

        System.out.println("Best Hidden layers " + bestHiddenlayers+ " Best learning rate " +  bestLearningRate + "  Best Momentum " + bestMomentum+ " Best Decay "+ bestDecay);
        return minError;
    }
}
